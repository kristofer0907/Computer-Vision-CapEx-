"""Drag the zone polygons around on one of your captures.

    python -m tools.edit_zones --image capture.jpg
    python -m tools.edit_zones --rgb mock            # grab a frame first
    python -m tools.edit_zones --image capture.jpg --reset   # start from defaults

Mouse:
    drag a corner        move that vertex
    drag inside a zone   move the whole zone
    click a zone         select it
    double click an edge add a vertex there
    right click a corner delete that vertex

Keys:
    TAB / 1-9   select the next / nth zone        arrows  nudge selection by 1 px
    a           add a zone (drawn as a rectangle) SHIFT+arrows  nudge by 10 px
    d           delete the selected zone          r  reset selected zone
    [ ]         shrink / grow selected zone       h  toggle help
    s           save to data/zones.json           q / ESC  quit without saving

You start from whatever zones already exist (data/zones.json, or the built-in
placeholder rectangles), so the normal flow is "drag five rectangles onto the
real bench" rather than tracing anything from scratch.

Everything is stored normalised to 0..1, so a saved layout survives a change of
capture resolution. The editor works in display pixels and converts on save.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from config import (DATA_DIR, GEOMETRY, STAGE_ORDER, ZONES, ZONES_FILE,
                    _DEFAULT_POLYGONS, ensure_dirs)
from pipeline.zones import px_to_mm

log = logging.getLogger("edit_zones")

WINDOW = "edit zones"
MAX_DISPLAY = (1500, 900)
GRAB_PX = 9                 # how close the cursor must be to grab a vertex

ACTIVE = (60, 168, 224)
IDLE = (110, 120, 135)
VERTEX = (250, 220, 90)
TEXT = (223, 227, 234)


# --------------------------------------------------------------------------
# Editor state. Deliberately free of any cv2 window calls so it can be driven
# from a test without a display.
# --------------------------------------------------------------------------
class ZoneEditor:
    """Polygons in image pixels, plus hit-testing and dragging."""

    def __init__(self, width: int, height: int,
                 polygons: dict[str, list[tuple[float, float]]]) -> None:
        self.w = width
        self.h = height
        self.zones: dict[str, list[list[float]]] = {
            name: [[x * width, y * height] for x, y in poly]
            for name, poly in polygons.items()
        }
        self.order = [n for n in STAGE_ORDER if n in self.zones]
        self.order += [n for n in self.zones if n not in self.order]
        self.selected: str | None = self.order[0] if self.order else None
        self.dirty = False
        self._drag: tuple[str, int | None, float, float] | None = None

    # ------------------------------------------------------------ selection
    def select(self, name: str | None) -> None:
        if name is None or name in self.zones:
            self.selected = name

    def select_next(self) -> None:
        if not self.order:
            return
        i = self.order.index(self.selected) if self.selected in self.order else -1
        self.selected = self.order[(i + 1) % len(self.order)]

    def select_index(self, n: int) -> None:
        if 0 <= n < len(self.order):
            self.selected = self.order[n]

    # ---------------------------------------------------------- hit testing
    def vertex_at(self, x: float, y: float, radius: float = GRAB_PX
                  ) -> tuple[str, int] | None:
        """Nearest vertex within `radius`, preferring the selected zone.

        Preferring the selection matters where two zones share an edge: the
        vertex you can see highlighted is the one you meant to grab.
        """
        best: tuple[str, int] | None = None
        best_d = radius
        names = ([self.selected] if self.selected else []) + \
                [n for n in self.order if n != self.selected]
        for name in names:
            for i, (vx, vy) in enumerate(self.zones.get(name, [])):
                d = math.hypot(vx - x, vy - y)
                if d <= best_d:
                    best, best_d = (name, i), d
            if best and name == self.selected:
                return best     # selected zone wins outright
        return best

    def zone_at(self, x: float, y: float) -> str | None:
        """Topmost zone containing the point, selected zone first."""
        names = ([self.selected] if self.selected else []) + \
                [n for n in self.order if n != self.selected]
        for name in names:
            poly = np.array(self.zones[name], np.float32)
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return name
        return None

    def edge_at(self, x: float, y: float, radius: float = GRAB_PX
                ) -> tuple[str, int] | None:
        """(zone, index to insert a new vertex at) for the nearest edge."""
        best = None
        best_d = radius
        for name, poly in self.zones.items():
            n = len(poly)
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                d = _point_segment_distance((x, y), a, b)
                if d < best_d:
                    best, best_d = (name, i + 1), d
        return best

    # --------------------------------------------------------------- edits
    def begin_drag(self, x: float, y: float) -> bool:
        """Start dragging a vertex, or a whole zone. False if nothing was hit."""
        hit = self.vertex_at(x, y)
        if hit:
            self.selected = hit[0]
            self._drag = (hit[0], hit[1], x, y)
            return True

        name = self.zone_at(x, y)
        if name:
            self.selected = name
            self._drag = (name, None, x, y)   # None index = move the whole zone
            return True

        self._drag = None
        return False

    def drag_to(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        name, index, px, py = self._drag
        dx, dy = x - px, y - py
        if index is None:
            for v in self.zones[name]:
                v[0] += dx
                v[1] += dy
        else:
            self.zones[name][index][0] += dx
            self.zones[name][index][1] += dy
        self._drag = (name, index, x, y)
        self.dirty = True
        self._clamp(name)

    def end_drag(self) -> None:
        self._drag = None

    @property
    def dragging(self) -> bool:
        return self._drag is not None

    def nudge(self, dx: float, dy: float) -> None:
        if not self.selected:
            return
        for v in self.zones[self.selected]:
            v[0] += dx
            v[1] += dy
        self.dirty = True
        self._clamp(self.selected)

    def scale_selected(self, factor: float) -> None:
        """Grow or shrink about the zone's own centre."""
        if not self.selected:
            return
        poly = self.zones[self.selected]
        cx = sum(v[0] for v in poly) / len(poly)
        cy = sum(v[1] for v in poly) / len(poly)
        for v in poly:
            v[0] = cx + (v[0] - cx) * factor
            v[1] = cy + (v[1] - cy) * factor
        self.dirty = True
        self._clamp(self.selected)

    def add_vertex(self, x: float, y: float) -> bool:
        hit = self.edge_at(x, y)
        if not hit:
            return False
        name, index = hit
        self.zones[name].insert(index, [x, y])
        self.selected = name
        self.dirty = True
        return True

    def delete_vertex(self, x: float, y: float) -> bool:
        hit = self.vertex_at(x, y)
        if not hit:
            return False
        name, index = hit
        if len(self.zones[name]) <= 3:
            log.warning("a zone needs at least 3 corners")
            return False
        self.zones[name].pop(index)
        self.dirty = True
        return True

    def add_zone(self, name: str) -> None:
        """A new zone as a rectangle in the middle, ready to be dragged."""
        if name in self.zones:
            log.warning("zone %r already exists", name)
            return
        w, h = self.w, self.h
        self.zones[name] = [[w * 0.40, h * 0.35], [w * 0.60, h * 0.35],
                            [w * 0.60, h * 0.65], [w * 0.40, h * 0.65]]
        self.order.append(name)
        self.selected = name
        self.dirty = True

    def delete_zone(self) -> None:
        if not self.selected:
            return
        self.zones.pop(self.selected, None)
        self.order = [n for n in self.order if n != self.selected]
        self.selected = self.order[0] if self.order else None
        self.dirty = True

    def reset_zone(self) -> None:
        """Put the selected zone back to its built-in placeholder rectangle."""
        if not self.selected or self.selected not in _DEFAULT_POLYGONS:
            log.warning("no default shape for %r", self.selected)
            return
        self.zones[self.selected] = [
            [x * self.w, y * self.h] for x, y in _DEFAULT_POLYGONS[self.selected]]
        self.dirty = True

    def _clamp(self, name: str) -> None:
        """Keep vertices inside the frame.

        A vertex dragged off-frame still saves as a valid normalised number
        outside 0..1, and then silently clips at a different place on a
        different capture resolution. Clamping here makes what you see be
        what gets stored.
        """
        for v in self.zones[name]:
            v[0] = min(max(v[0], 0.0), self.w - 1)
            v[1] = min(max(v[1], 0.0), self.h - 1)

    # -------------------------------------------------------------- output
    def normalised(self) -> dict[str, list[list[float]]]:
        return {name: [[round(x / self.w, 5), round(y / self.h, 5)]
                       for x, y in poly]
                for name, poly in self.zones.items()}

    def summary(self) -> list[str]:
        out = []
        for name in self.order:
            poly = self.zones[name]
            xs = [v[0] for v in poly]
            ys = [v[1] for v in poly]
            out.append(f"{name:<10} {len(poly)} pts  "
                       f"{px_to_mm(max(xs) - min(xs)):5.0f} x "
                       f"{px_to_mm(max(ys) - min(ys)):4.0f} mm")
        return out

    def overlaps(self, min_fraction: float = 0.02) -> list[tuple[str, str, float]]:
        """Pairs of zones overlapping by more than `min_fraction` of the smaller.

        Not fatal - zone_at resolves ties by process order - but a real
        overlap is almost always a mis-drag, and a vial inside it gets
        whichever stage comes first rather than the one you meant.

        The threshold exists because adjacent zones legitimately share an
        edge, and a shared edge is "inside" both polygons as far as
        pointPolygonTest is concerned. Warning on that would mean nagging on
        every single save of a correct layout, which trains you to ignore the
        warning that matters.
        """
        found = []
        masks = {}
        for name, poly in self.zones.items():
            m = np.zeros((self.h, self.w), np.uint8)
            cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
            masks[name] = m
        names = list(masks)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = int((masks[a] & masks[b]).sum())
                if not shared:
                    continue
                smaller = min(int(masks[a].sum()), int(masks[b].sum())) or 1
                fraction = shared / smaller
                if fraction >= min_fraction:
                    found.append((a, b, fraction))
        return found


def _point_segment_distance(p, a, b) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------
HELP = [
    "drag corner: move vertex     drag inside: move zone",
    "dbl-click edge: add vertex   right-click corner: delete vertex",
    "TAB/1-9 select   a add zone   d delete zone   r reset   [ ] resize",
    "arrows nudge (SHIFT x10)     s save     q quit     h hide help",
]


class EditorWindow:
    """Owns the display scaling and the cv2 event loop.

    The image is scaled to fit the screen once, and every mouse coordinate is
    divided back into image pixels. Doing it this way rather than with
    WINDOW_NORMAL avoids the long-standing difference between OpenCV builds
    in whether callback coordinates are window-space or image-space.
    """

    def __init__(self, image: np.ndarray, editor: ZoneEditor) -> None:
        self.image = image
        self.editor = editor
        h, w = image.shape[:2]
        self.scale = min(MAX_DISPLAY[0] / w, MAX_DISPLAY[1] / h, 1.0)
        self.display = (cv2.resize(image, (int(w * self.scale), int(h * self.scale)))
                        if self.scale < 1.0 else image.copy())
        self.show_help = True
        self.cursor = (0.0, 0.0)

    def to_image(self, x: int, y: int) -> tuple[float, float]:
        return x / self.scale, y / self.scale

    def on_mouse(self, event, x, y, flags, _param) -> None:
        ix, iy = self.to_image(x, y)
        self.cursor = (ix, iy)

        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.editor.add_vertex(ix, iy)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.editor.begin_drag(ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            self.editor.drag_to(ix, iy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.editor.end_drag()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.editor.delete_vertex(ix, iy)

    def render(self) -> np.ndarray:
        out = self.display.copy()
        s = self.scale

        for name in self.editor.order:
            poly = self.editor.zones[name]
            pts = np.array([[v[0] * s, v[1] * s] for v in poly], np.int32)
            active = name == self.editor.selected
            colour = ACTIVE if active else IDLE

            if active:
                # Tint the selected zone so it is obvious which one arrows move.
                overlay = out.copy()
                cv2.fillPoly(overlay, [pts], colour)
                cv2.addWeighted(overlay, 0.14, out, 0.86, 0, out)

            cv2.polylines(out, [pts], True, colour, 2 if active else 1, cv2.LINE_AA)
            if active:
                for px, py in pts:
                    cv2.circle(out, (int(px), int(py)), 5, VERTEX, -1, cv2.LINE_AA)
                    cv2.circle(out, (int(px), int(py)), 5, (30, 30, 30), 1, cv2.LINE_AA)

            label = pts.min(axis=0)
            cv2.putText(out, name, (int(label[0]) + 5, int(label[1]) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        self._draw_banner(out)
        return out

    def _draw_banner(self, out: np.ndarray) -> None:
        lines = []
        sel = self.editor.selected
        if sel:
            poly = self.editor.zones[sel]
            xs = [v[0] for v in poly]
            ys = [v[1] for v in poly]
            lines.append(f"[{sel}]  {len(poly)} corners  "
                         f"{px_to_mm(max(xs) - min(xs)):.0f} x "
                         f"{px_to_mm(max(ys) - min(ys)):.0f} mm"
                         f"{'   UNSAVED' if self.editor.dirty else ''}")
        else:
            lines.append("no zones - press 'a' to add one")
        lines.append(f"cursor {self.cursor[0]:.0f},{self.cursor[1]:.0f} px")
        if self.show_help:
            lines += HELP

        pad = 6
        height = 18 * len(lines) + pad
        cv2.rectangle(out, (0, 0), (out.shape[1], height), (18, 20, 24), -1)
        for i, text in enumerate(lines):
            colour = TEXT if i < 2 else (140, 148, 160)
            cv2.putText(out, text, (8, 15 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, colour, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ loop
    def run(self) -> bool:
        """True if the user asked to save."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self.render())
                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue
                action = self.handle_key(key)
                if action in ("save", "quit"):
                    return action == "save"
        finally:
            cv2.destroyAllWindows()

    def handle_key(self, key: int) -> str | None:
        ed = self.editor
        k = key & 0xFF

        if k in (ord("q"), 27):
            return "quit"
        if k == ord("s"):
            return "save"
        if k == ord("\t"):
            ed.select_next()
        elif ord("1") <= k <= ord("9"):
            ed.select_index(k - ord("1"))
        elif k == ord("h"):
            self.show_help = not self.show_help
        elif k == ord("a"):
            ed.add_zone(_next_zone_name(ed))
        elif k == ord("d"):
            ed.delete_zone()
        elif k == ord("r"):
            ed.reset_zone()
        elif k == ord("["):
            ed.scale_selected(0.96)
        elif k == ord("]"):
            ed.scale_selected(1.04)
        else:
            # Arrow keys differ between builds and platforms, so match on the
            # full key code rather than the masked byte.
            step = 10.0 if key & 0x10000 else 1.0
            arrows = {81: (-1, 0), 82: (0, -1), 83: (1, 0), 84: (0, 1),
                      2424832: (-1, 0), 2490368: (0, -1),
                      2555904: (1, 0), 2621440: (0, 1)}
            for code, (dx, dy) in arrows.items():
                if key in (code, code | 0x10000):
                    ed.nudge(dx * step, dy * step)
                    break
        return None


def _next_zone_name(editor: ZoneEditor) -> str:
    for name in STAGE_ORDER:
        if name != "oven" and name not in editor.zones:
            return name
    i = 1
    while f"zone_{i}" in editor.zones:
        i += 1
    return f"zone_{i}"


# --------------------------------------------------------------------------
def load_image(image: str | None, backend: str | None) -> np.ndarray:
    if image:
        img = cv2.imread(image)
        if img is None:
            raise SystemExit(f"could not read {image}")
        return img

    from drivers.rgb_cam import create_camera
    camera = create_camera(backend)
    try:
        for _ in range(4):      # let exposure settle
            frame = camera.capture()
        return frame.image.copy()
    finally:
        camera.stop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="the capture to edit against")
    p.add_argument("--rgb", default=None, help="camera backend if no --image")
    p.add_argument("--reset", action="store_true",
                   help="start from the placeholder rectangles, "
                        "ignoring any saved zones")
    p.add_argument("--out", default=str(ZONES_FILE))
    p.add_argument("--save-frame", help="write the grabbed frame here and exit")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    ensure_dirs()

    image = load_image(args.image, args.rgb)
    if args.save_frame:
        cv2.imwrite(args.save_frame, image)
        log.info("wrote %s - copy it somewhere with a display and rerun "
                 "with --image", args.save_frame)
        return 0

    polygons = dict(_DEFAULT_POLYGONS) if args.reset else dict(ZONES.polygons)
    h, w = image.shape[:2]
    editor = ZoneEditor(w, h, polygons)
    log.info("editing %d zones on a %dx%d image (%.3f mm/px)",
             len(editor.zones), w, h, px_to_mm(1.0))
    log.info("starting from %s",
             "built-in placeholders" if (args.reset or not ZONES.calibrated)
             else args.out)

    window = EditorWindow(image, editor)
    if window.scale < 1.0:
        log.info("display scaled to %.0f%% to fit the screen", window.scale * 100)

    try:
        should_save = window.run()
    except cv2.error as exc:
        log.error("cannot open a window: %s",
                  str(exc).strip().splitlines()[-1][:120])
        log.error("this tool needs a display. Over SSH use `ssh -X`, or grab a "
                  "frame with --save-frame and edit it on your laptop.")
        return 2

    if not should_save:
        log.info("quit without saving")
        return 1

    return save(editor, Path(args.out))


def save(editor: ZoneEditor, out: Path) -> int:
    for a, b, fraction in editor.overlaps():
        log.warning("zones %s and %s overlap by %.0f%% - a vial in the overlap "
                    "will be assigned to whichever comes first in the process "
                    "order", a, b, fraction * 100)
    missing = [s for s in STAGE_ORDER if s != "oven" and s not in editor.zones]
    if missing:
        log.warning("no polygon for %s - a vial there will be reported unstaged",
                    ", ".join(missing))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(editor.normalised(), indent=2))
    log.info("wrote %d zones to %s", len(editor.zones), out)
    for line in editor.summary():
        log.info("  %s", line)
    log.info("total span %.0f mm (platform is %.0f mm)",
             _total_span_mm(editor), GEOMETRY.platform_length_mm)
    log.info("restart the monitor for this to take effect")
    return 0


def _total_span_mm(editor: ZoneEditor) -> float:
    xs = [v[0] for poly in editor.zones.values() for v in poly]
    return px_to_mm(max(xs) - min(xs)) if xs else 0.0


if __name__ == "__main__":
    sys.exit(main())
