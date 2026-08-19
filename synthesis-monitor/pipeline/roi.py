"""Region-of-interest extraction: turning a centroid into pixels to look at.

Pure geometry and array slicing. No thresholds, no decisions - this module
hands the detectors and the feature extractor the pixels they asked for and
has no opinion about what is in them.

Two shapes of ROI exist because the detection plans need both:

  * per-vial crops, for turbidity / sol-gel / colour change,
  * per-zone crops, for spill and overflow, where the interesting surface is
    the aluminium bench or the filter paper *between* the vials, not a vial.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import DETECTION


def clip_box(x0: int, y0: int, x1: int, y1: int, shape: tuple[int, ...]
             ) -> tuple[int, int, int, int]:
    """Clamp a box to the image, keeping it non-empty where possible."""
    h, w = shape[0], shape[1]
    return (max(0, min(x0, w - 1)), max(0, min(y0, h - 1)),
            max(1, min(x1, w)), max(1, min(y1, h)))


def square_box(cx: float, cy: float, radius: float,
               scale: float | None = None) -> tuple[int, int, int, int]:
    """Square box of side `scale` * radius centred on (cx, cy), unclipped."""
    s = DETECTION.roi_scale if scale is None else scale
    half = radius * s * 0.5
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def crop(image: np.ndarray, cx: float, cy: float, radius: float,
         scale: float | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop around a vial. Returns (view, clipped box).

    The returned array is a *view* into `image`, not a copy - cheap, but it
    means writing to it edits the frame. Call .copy() before drawing on it.
    A vial near the frame edge yields a smaller-than-nominal crop rather than
    a padded one, so anything that compares two crops must not assume they
    are the same size.
    """
    box = clip_box(*square_box(cx, cy, radius, scale), image.shape)
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1], box


def disc_mask(shape: tuple[int, int], cx: float, cy: float, radius: float,
              shrink_px: float = 2.0) -> np.ndarray:
    """uint8 mask, 255 inside the vial's liquid disc.

    `shrink_px` pulls the mask inside the glass rim. Sampling the rim itself
    would mix the glass wall, the meniscus and any surviving specular
    highlight into the liquid statistics; cross-polarisation reduces that
    highlight but does not remove it.
    """
    m = np.zeros(shape[:2], np.uint8)
    r = max(1, int(round(radius - shrink_px)))
    cv2.circle(m, (int(round(cx)), int(round(cy))), r, 255, -1)
    return m


def crop_with_mask(image: np.ndarray, cx: float, cy: float, radius: float,
                   scale: float | None = None
                   ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """(crop copy, disc mask in crop coordinates, box in frame coordinates)."""
    view, box = crop(image, cx, cy, radius, scale)
    x0, y0, _, _ = box
    mask = disc_mask(view.shape, cx - x0, cy - y0, radius)
    return view.copy(), mask, box


def zone_crop(image: np.ndarray, bounds: tuple[int, int, int, int]
              ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a whole zone by its bounding box, clipped to the frame."""
    box = clip_box(*bounds, image.shape)
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1], box


def exclude_discs(mask: np.ndarray, centers: list[tuple[float, float, float]],
                  origin: tuple[int, int] = (0, 0),
                  pad_px: float = 3.0) -> np.ndarray:
    """Punch the vials out of a zone mask, leaving only the bench surface.

    This is what a spill check wants: the aluminium (or the filter paper) with
    the vials that legitimately sit on it removed, so a vial's own colour
    cannot be mistaken for a wet patch. `centers` is (cx, cy, radius) in frame
    coordinates; `origin` is the crop's top-left in the same frame.
    """
    out = mask.copy()
    ox, oy = origin
    for cx, cy, r in centers:
        cv2.circle(out, (int(round(cx - ox)), int(round(cy - oy))),
                   int(round(r + pad_px)), 0, -1)
    return out


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resize `image` to `reference`'s height and width.

    Needed before differencing two crops of the same vial from different
    frames: the crops are clipped independently, so a vial that moved towards
    the frame edge produces a smaller box and a raw cv2.absdiff would throw.
    """
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]),
                      interpolation=cv2.INTER_AREA)
