"""Render the geometry the pipeline is actually using, so you can check it by eye.

    python -m tools.inspect_roi                      # simulator frame
    python -m tools.inspect_roi --image cal.jpg      # one of your captures
    python -m tools.inspect_roi --images captures/   # every image in a folder
    python -m tools.inspect_roi --zone heating       # zoom one zone
    python -m tools.inspect_roi --track 3            # zoom one vial

Vials on a real capture come from the localiser, which by default means the
hand marks in data/vials.json (see tools/mark_vials.py). Without them a real
image has no vials to crop and the bench mask is just the zone polygon, so
mark them once before expecting the per-vial panels to say anything.

Writes a contact sheet to data/inspect/ and prints the numbers alongside it.

Masks and polygons are the part of this system that unit tests are worst at.
An assertion can tell you a mask is non-empty; it cannot tell you the disc is
sitting two pixels off the rim, or that a zone polygon clips the corner of the
heater pad. Both of those are obvious in one glance at an image and invisible
in a passing test, so this exists alongside the tests rather than instead of
them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from config import DATA_DIR, DETECTION, ensure_dirs
from pipeline import roi
from pipeline.zones import ZoneMap, px_to_mm

log = logging.getLogger("inspect")

OUT_DIR = DATA_DIR / "inspect"
LABEL_BGR = (223, 227, 234)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    """Caption a panel so the contact sheet is readable without a legend."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (20, 22, 26), -1)
    cv2.putText(out, text, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                LABEL_BGR, 1, cv2.LINE_AA)
    return out


def _tile(panels: list[np.ndarray], width: int = 260) -> np.ndarray:
    """Lay panels out in a row, scaled to a common width."""
    scaled = []
    for p in panels:
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        h = max(1, int(p.shape[0] * width / max(p.shape[1], 1)))
        scaled.append(cv2.resize(p, (width, h), interpolation=cv2.INTER_NEAREST))
    tallest = max(s.shape[0] for s in scaled)
    padded = [cv2.copyMakeBorder(s, 0, tallest - s.shape[0], 0, 4,
                                 cv2.BORDER_CONSTANT, value=(20, 22, 26))
              for s in scaled]
    return np.hstack(padded)


def grab(image_path: str | None, images_dir: str | None, backend: str | None):
    """Yield Frames from a file, a folder, or the simulator.

    Everything goes through a real Frame rather than a bare array, so the
    localiser sees exactly what it sees in the running pipeline - including
    the source filename, which is how ManualLocalizer finds the right marks.
    """
    from drivers.rgb_cam import FileCameraSource, create_camera

    if image_path or images_dir:
        source = FileCameraSource(image_path or images_dir, loop=False)
        source.start()
        try:
            while True:
                try:
                    yield source.capture()
                except StopIteration:
                    return
        finally:
            source.stop()
        return

    camera = create_camera(backend)
    try:
        for _ in range(6):          # let the scene advance past the first frame
            frame = camera.capture()
        yield frame
    finally:
        camera.stop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", help="one of your captures")
    src.add_argument("--images", help="a folder of captures")
    p.add_argument("--rgb", default="mock", help="camera backend when capturing")
    p.add_argument("--localizer", default="auto",
                   help="where vials come from: auto | manual | ground_truth | null")
    p.add_argument("--zone", help="render this zone's masks in detail")
    p.add_argument("--track", type=int, help="render this vial's index in detail")
    p.add_argument("--scale", type=float, default=DETECTION.roi_scale,
                   help="ROI size as a multiple of vial radius")
    p.add_argument("--all-vials", action="store_true",
                   help="contact sheet of every vial's crop and mask")
    p.add_argument("--show", action="store_true",
                   help="open a window as well as writing files "
                        "(needs a display; over SSH use ssh -X)")
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    ensure_dirs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.localize import create_localizer

    localizer = create_localizer(args.localizer)
    try:
        localizer.start()
    except FileNotFoundError as exc:
        # Marks missing is the common case on a first run against real
        # images. Say what to do about it rather than dying on a traceback.
        log.error("%s", exc)
        return 2

    written: list[Path] = []
    try:
        for n, frame in enumerate(grab(args.image, args.images, args.rgb)):
            written += _inspect_one(frame, localizer, args, out_dir,
                                    multi=bool(args.images), index=n)
    finally:
        localizer.stop()

    if not written:
        log.error("no images inspected")
        return 1
    for path in written:
        log.info("wrote %s", path)
    return 0


def _inspect_one(frame, localizer, args, out_dir: Path, multi: bool,
                 index: int) -> list[Path]:
    """Render the contact sheet for one frame."""
    image = frame.image
    h, w = image.shape[:2]
    zones = ZoneMap(w, h)
    name = str((frame.truth or {}).get("name", "") or f"frame_{frame.frame_id}")
    stem = f"{index:03d}_" if multi else ""

    log.info("")
    log.info("%s  %dx%d  %.3f mm/px  zones: %s", name, w, h, px_to_mm(1.0),
             ", ".join(zones.names) or "none configured")

    detections = localizer.locate(frame)
    if not detections:
        log.warning("  no vials from localiser %r - per-vial panels skipped. "
                    "Mark them with `python -m tools.mark_vials --image <file>`",
                    localizer.name)

    written: list[Path] = []

    # ---------------------------------------------------------------- zones
    overview = zones.draw(image)
    for i, d in enumerate(detections):
        cv2.circle(overview, (int(d.cx), int(d.cy)), int(d.radius),
                   (110, 199, 98), 1, cv2.LINE_AA)
        cv2.putText(overview, str(i), (int(d.cx) + 6, int(d.cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 199, 98), 1, cv2.LINE_AA)
    written.append(_write(out_dir / f"{stem}zones.jpg", overview))

    for zone in zones.names:
        x0, y0, x1, y1 = zones.bounds(zone)
        inside = sum(1 for d in detections if x0 <= d.cx < x1 and y0 <= d.cy < y1)
        log.info("  %-9s bounds=(%4d,%3d)-(%4d,%3d)  %6.0f x %5.0f mm  "
                 "%6d px  %d vials", zone, x0, y0, x1, y1,
                 px_to_mm(x1 - x0), px_to_mm(y1 - y0),
                 int(zones.mask(zone).sum() // 255), inside)

    # ------------------------------------------------------------- one vial
    if detections:
        pick = args.track if args.track is not None else 0
        target = detections[pick] if 0 <= pick < len(detections) else detections[0]
        crop, mask, box = roi.crop_with_mask(image, target.cx, target.cy,
                                             target.radius, args.scale)
        if crop.size:
            masked = cv2.bitwise_and(crop, crop, mask=mask)
            b, g, r_, _ = cv2.mean(crop, mask=mask)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hm, sm, vm, _ = cv2.mean(hsv, mask=mask)
            log.info("  vial %d: centre=(%.0f,%.0f) r=%.1f px (%.1f mm dia) "
                     "crop=%dx%d disc=%d px",
                     pick, target.cx, target.cy, target.radius,
                     px_to_mm(target.radius) * 2, crop.shape[1], crop.shape[0],
                     int(mask.sum() // 255))
            log.info("    mean inside disc: BGR=(%.0f,%.0f,%.0f)  "
                     "HSV=(%.0f,%.0f,%.0f)", b, g, r_, hm, sm, vm)
            written.append(_write(out_dir / f"{stem}vial.jpg", _tile([
                _label(crop, f"crop (vial {pick})"),
                _label(mask, "disc mask"),
                _label(masked, "masked liquid"),
            ], width=180)))

    # ------------------------------------------------------------- one zone
    zone_name = args.zone or (zones.names[0] if zones.names else None)
    if zone_name and zone_name in zones.polygons_px:
        crop, box = roi.zone_crop(image, zones.bounds(zone_name))
        x0, y0, x1, y1 = box
        zone_mask = zones.mask(zone_name)[y0:y1, x0:x1]
        inside = [(d.cx, d.cy, d.radius) for d in detections
                  if x0 <= d.cx < x1 and y0 <= d.cy < y1]
        bench = roi.exclude_discs(zone_mask, inside, (x0, y0))
        log.info("  zone %s: %d vials inside, polygon=%d px, bench=%d px "
                 "(%.0f%% bare surface)", zone_name, len(inside),
                 int(zone_mask.sum() // 255), int(bench.sum() // 255),
                 100.0 * bench.sum() / max(zone_mask.sum(), 1))
        written.append(_write(out_dir / f"{stem}zone.jpg", _tile([
            _label(crop, f"{zone_name} crop"),
            _label(zone_mask, "polygon mask"),
            _label(bench, "bench mask (vials removed)"),
            _label(cv2.bitwise_and(crop, crop, mask=bench), "bench pixels"),
        ])))

    # -------------------------------------------------------- every vial
    if args.all_vials and detections:
        sheet = _vial_sheet(image, detections, args.scale)
        if sheet is not None:
            written.append(_write(out_dir / f"{stem}vials_all.jpg", sheet))

    if args.show:
        _show(written, name)

    return written


def _vial_sheet(image: np.ndarray, detections, scale: float,
                per_row: int = 9) -> np.ndarray | None:
    """Every vial's crop over its disc mask, in one grid.

    This is the view for tuning DETECTION.roi_scale: too small and the crop
    clips the rim, too large and it swallows the neighbouring vial. Both are
    obvious across 18 tiles at once and easy to miss on one.
    """
    tiles = []
    for i, d in enumerate(detections):
        crop, mask, _ = roi.crop_with_mask(image, d.cx, d.cy, d.radius, scale)
        if crop.size == 0:
            continue
        edge = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        edge = cv2.Canny(mask, 50, 150)
        marked = crop.copy()
        marked[edge > 0] = (90, 220, 250)      # disc boundary over the pixels
        tiles.append(_label(marked, f"{i}"))
    if not tiles:
        return None

    side = max(max(t.shape[:2]) for t in tiles)
    square = [cv2.copyMakeBorder(t, 0, side - t.shape[0], 0, side - t.shape[1],
                                 cv2.BORDER_CONSTANT, value=(20, 22, 26))
              for t in tiles]
    rows = []
    for i in range(0, len(square), per_row):
        chunk = square[i:i + per_row]
        while len(chunk) < per_row:
            chunk.append(np.full_like(square[0], 24))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)
    return cv2.resize(grid, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)


def _show(paths: list[Path], title: str) -> None:
    """Display what was just written. Silently skips if there is no display."""
    try:
        for path in paths:
            img = cv2.imread(str(path))
            if img is not None:
                cv2.imshow(f"{title} - {path.stem}", img)
        log.info("  press any key in a window to continue, q to stop showing")
        if (cv2.waitKey(0) & 0xFF) == ord("q"):
            raise KeyboardInterrupt
        cv2.destroyAllWindows()
    except cv2.error as exc:
        # No display: this is normal on the Pi over plain SSH and is not
        # worth failing over - the files are already on disk.
        log.warning("cannot open a window (%s) - the images are still written",
                    str(exc).strip().splitlines()[-1][:80])


def _write(path: Path, image: np.ndarray) -> Path:
    cv2.imwrite(str(path), image)
    return path


if __name__ == "__main__":
    sys.exit(main())
