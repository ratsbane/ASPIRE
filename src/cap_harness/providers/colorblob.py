"""Text-prompted segmentation by colour, for stations without a model server.

A fallback :class:`~cap_harness.protocols.SegmentationProvider`: the query's
colour word selects a chromatic-difference mask (thresholds measured on the
reference bench's wooden blocks across lighting changes), and each connected
component becomes one scored segmentation. It knows nothing but colour -- a
"blue mug" and a "blue block" are the same thing to it -- which is exactly the
gap a real segmentation model closes. Its value is that the full
localization pipeline (mask, point cloud, geometry) runs end to end on a bench
that has not yet earned a GPU model service, and every run records which
provider produced its masks.
"""

from __future__ import annotations

import numpy as np

from cap_harness.contracts import CameraObservation, Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode

MIN_COMPONENT_PX = 150
MAX_COMPONENTS = 4

#: Chromatic difference rules measured on the reference bench (wooden blocks,
#: mixed daylight/lamp): each colour is a set of channel margins that survived
#: lighting changes which defeated absolute HSV bounds.
_RULES = {
    "green": lambda r, g, b: (g > r + 15) & (g > b + 15),
    "red": lambda r, g, b: (r > g + 30) & (r > b + 30),
    "blue": lambda r, g, b: (b > r + 20) & (b > g + 10),
}


def _colour_of(query: str) -> str | None:
    words = query.lower().split()
    return next((w for w in words if w in _RULES), None)


class ColorBlobSegmentationProvider:
    """Segment by colour word; one segmentation per connected component."""

    def segment_text(self, observation: CameraObservation, query: str) -> SegmentationSet:
        colour = _colour_of(query)
        if colour is None:
            return SegmentationSet(
                ok=False,
                error=ApiError(
                    code=ErrorCode.NO_SEGMENTATION,
                    message=(
                        f"the colour-blob provider found no known colour in {query!r}; "
                        f"it only knows {sorted(_RULES)}"
                    ),
                ),
            )
        rgb = np.asarray(observation.rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
        saturation = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        mask = _RULES[colour](r, g, b) & (saturation > 25)

        from scipy import ndimage

        labels, count = ndimage.label(mask)
        sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, count + 1))
        keep = [
            (int(size), index + 1)
            for index, size in enumerate(sizes)
            if size >= MIN_COMPONENT_PX
        ]
        if not keep:
            return SegmentationSet(
                ok=False,
                error=ApiError(
                    code=ErrorCode.NO_SEGMENTATION,
                    message=f"no {colour} region of at least {MIN_COMPONENT_PX} px in view",
                ),
                diagnostics={"provider": "colorblob", "raw_mask_px": int(mask.sum())},
            )
        keep.sort(reverse=True)
        largest = keep[0][0]
        segmentations = [
            Segmentation(
                mask=labels == index,
                label=f"{colour} region",
                # A heuristic's confidence should look like one: proportional
                # prominence, never near 1.0.
                score=min(0.8, 0.4 + 0.4 * size / largest),
                camera_name=observation.frame,
                frame=observation.frame,
            )
            for size, index in keep[:MAX_COMPONENTS]
        ]
        return SegmentationSet(
            ok=True,
            segmentations=segmentations,
            diagnostics={"provider": "colorblob", "components": len(keep)},
        )

    def segment_points(
        self,
        observation: CameraObservation,
        points_px: np.ndarray,
        *,
        point_labels: np.ndarray | None = None,
    ) -> SegmentationSet:
        return SegmentationSet(
            ok=False,
            error=ApiError(
                code=ErrorCode.UNSUPPORTED,
                message="the colour-blob provider segments by colour text only",
            ),
        )


__all__ = ["ColorBlobSegmentationProvider"]
