"""Shoot a folder of stills at a fixed interval.

    python -m tools.capture_set --out data/captures/localization --interval 1 --count 300
    python -m tools.capture_set --out data/captures/flatfield --count 10 --full
    python -m tools.capture_set --out data/captures/test --rgb mock --count 5

This is a dataset tool, not part of the runtime. It writes lossless PNGs plus a
session.json recording every control the sensor was actually running under, so
two sets shot weeks apart can be checked for comparability instead of assumed
to be comparable.

Manual settings, and which of them software can even reach:

    exposure, gain, white balance   sensor side - set here, see --exposure etc.
    focus, focal length, aperture   mechanical rings on the Arducam varifocal.
                                    Set them by hand, then do not touch them
                                    again for the life of the dataset.

Defaults lock exposure and white balance because the pipeline compares vials
across frames and across sessions. Leaving auto-exposure and auto-white-balance
on lets the camera re-decide what "white" means between two frames, which shows
up later as a colour anomaly that no chemistry caused. Pass --auto to override,
but only for a viewfinding shot you are going to throw away.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from drivers.base import HardwareUnavailable
from drivers.rgb_cam import PiCameraSource, create_camera

log = logging.getLogger("capture_set")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture a folder of stills at a fixed interval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--out", required=True, type=Path,
                   help="output directory; created if missing")
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between frames")
    p.add_argument("--count", type=int, default=0,
                   help="number of frames; 0 means run until Ctrl-C")
    p.add_argument("--prefix", default="frame", help="filename prefix")
    p.add_argument("--jpeg", action="store_true",
                   help="write JPEGs instead of lossless PNG")
    p.add_argument("--note", default=None,
                   help="free text recorded in session.json, e.g. 'lids off, 12 vials'")

    cam = p.add_argument_group("camera")
    cam.add_argument("--rgb", default="auto",
                     choices=["auto", "picamera2", "mock", "file"])
    cam.add_argument("--width", type=int, default=None)
    cam.add_argument("--height", type=int, default=None)
    cam.add_argument("--full", action="store_true",
                     help="full sensor, 4056x3040 (overrides --width/--height)")

    ctl = p.add_argument_group("sensor controls (picamera2 only)")
    ctl.add_argument("--exposure", type=int, default=20000,
                     help="microseconds; keep to multiples of 10000 against "
                          "100 Hz mains flicker")
    ctl.add_argument("--gain", type=float, default=1.0,
                     help="analogue gain; raise the light, not this")
    ctl.add_argument("--awb-gains", default=None, metavar="R,B",
                     help="fixed colour gains, e.g. 1.8,1.6. Omit to let AWB "
                          "settle once and then freeze whatever it chose.")
    ctl.add_argument("--auto", action="store_true",
                     help="leave AE/AWB running. For throwaway framing shots only.")

    return p.parse_args(argv)


def build_controls(args: argparse.Namespace) -> dict:
    """The control dict for a locked sensor. Empty when --auto."""
    if args.auto:
        return {}
    controls: dict = {
        "AeEnable": False,
        "ExposureTime": args.exposure,
        "AnalogueGain": args.gain,
        # A frame can never be shorter than its own exposure; give libcamera a
        # ceiling well above it so a long exposure is not silently clipped.
        "FrameDurationLimits": (max(args.exposure, 33333), 2_000_000),
    }
    if args.awb_gains:
        r, b = (float(v) for v in args.awb_gains.split(","))
        controls["AwbEnable"] = False
        controls["ColourGains"] = (r, b)
    return controls


def freeze_awb(cam) -> tuple[float, float] | None:
    """Let AWB choose once, read its answer, then switch it off and pin it.

    Without this the camera re-decides white between frames. With it the whole
    session shares one white point - and the numbers are printed so the next
    session can be pinned to the same one with --awb-gains.
    """
    meta = cam.apply_controls({"AwbEnable": True})
    gains = meta.get("ColourGains")
    if not gains:
        log.warning("camera reported no ColourGains; AWB left as-is")
        return None
    r, b = float(gains[0]), float(gains[1])
    cam.apply_controls({"AwbEnable": False, "ColourGains": (r, b)})
    return r, b


def open_camera(backend: str, width: int | None, height: int | None):
    """Start a camera, honouring an explicit capture size when one is given.

    create_camera() deliberately takes no size - the runtime always wants the
    configured one. A dataset does not: calibration and pixel-density work want
    the full sensor, which is far too slow to run a 45 s pipeline on. So a size
    request builds the Pi source directly and keeps auto's fallback behaviour.
    """
    if width is None and height is None:
        return create_camera(backend)
    if backend == "mock":
        log.warning("mock camera renders at its configured size; --width ignored")
        return create_camera(backend)
    try:
        cam = PiCameraSource(width, height)
        cam.start()
        return cam
    except HardwareUnavailable as exc:
        if backend == "picamera2":
            raise
        log.warning("RGB hardware unavailable (%s) - falling back", exc)
        return create_camera("mock")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    width, height = args.width, args.height
    if args.full:
        width, height = 4056, 3040

    cam = open_camera(args.rgb, width, height)

    controls_used: dict = {}
    if hasattr(cam, "apply_controls"):
        awb = None
        if not args.auto and not args.awb_gains:
            awb = freeze_awb(cam)
            if awb:
                log.info("AWB frozen at ColourGains %.3f,%.3f "
                         "- pass --awb-gains %.3f,%.3f next time", *awb, *awb)
        meta = cam.apply_controls(build_controls(args))
        controls_used = {
            "requested": {k: v for k, v in build_controls(args).items()},
            "actual_exposure_us": meta.get("ExposureTime"),
            "actual_gain": meta.get("AnalogueGain"),
            "actual_colour_gains": meta.get("ColourGains"),
        }
        log.info("sensor locked: %s us, gain %s, gains %s",
                 meta.get("ExposureTime"), meta.get("AnalogueGain"),
                 meta.get("ColourGains"))
    elif not args.auto:
        log.warning("%s has no controllable sensor; exposure flags ignored",
                    cam.name)

    args.out.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if args.jpeg else "png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if args.jpeg else []

    frames: list[dict] = []
    started = time.time()
    log.info("capturing to %s every %.2fs (%s) - Ctrl-C to stop",
             args.out, args.interval, args.count or "unlimited")

    try:
        while not args.count or len(frames) < args.count:
            t0 = time.monotonic()
            frame = cam.capture()
            name = f"{args.prefix}_{len(frames):05d}.{ext}"
            cv2.imwrite(str(args.out / name), frame.image, params)
            frames.append({"file": name, "timestamp": frame.timestamp,
                           "frame_id": frame.frame_id})
            if len(frames) % 10 == 0 or len(frames) == 1:
                log.info("  %d frames", len(frames))
            remaining = args.interval - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        cam.stop()

    session = {
        "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "source": cam.name,
        "simulated": cam.simulated,
        "interval_s": args.interval,
        "format": ext,
        "note": args.note,
        "controls": controls_used or "auto",
        "frame_count": len(frames),
        "frames": frames,
    }
    (args.out / "session.json").write_text(json.dumps(session, indent=2))
    log.info("wrote %d frames + session.json to %s", len(frames), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
