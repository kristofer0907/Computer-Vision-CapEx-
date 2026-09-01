"""Find the camera's intrinsic matrix + lens distortion from checkerboard photos,
and undistort future images with the result.

Calibrate once (finds capture/second_iteration/calibration_imgs by default):

    python -m tools.calibrate_camera

Undistort a new image using the saved calibration:

    python -m tools.calibrate_camera --undistort path/to/image.jpg

The checkerboard used has 7x9 squares, which means 6x8 INNER corners -- that
is what OpenCV actually looks for. If you print a different board, count the
inner corners (where 4 squares touch) and pass --cols/--rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import DATA_DIR

DEFAULT_IMAGES_DIR = Path("capture/second_iteration/calibration_imgs")
CALIBRATION_FILE = DATA_DIR / "camera_calibration.npz"

# Inner corners of the checkerboard (columns, rows), not squares.
BOARD_COLS = 6
BOARD_ROWS = 8


def find_corners(images_dir: Path, cols: int, rows: int):
    """Look for the checkerboard in every image in images_dir.

    Returns (object_points, image_points, image_size) ready for cv2.calibrateCamera.
    """
    # One "object point" set shared by every photo: the checkerboard corners
    # in a flat grid, e.g. (0,0,0), (1,0,0), (2,0,0), ... We don't know the
    # real square size in mm, so we just count in "squares" -- that's enough
    # to compute the lens distortion and camera matrix correctly.
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    object_points = []  # the same grid, once per image that worked
    image_points = []   # the corners actually found, once per image
    image_size = None

    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} images in {images_dir}")

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip {path.name}: could not read file")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height)

        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not found:
            print(f"  skip {path.name}: no checkerboard found")
            continue

        # Refine corner locations to sub-pixel accuracy for a better fit.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        object_points.append(grid)
        image_points.append(corners)
        print(f"  ok   {path.name}")

    print(f"Used {len(object_points)} / {len(image_paths)} images")
    return object_points, image_points, image_size


def calibrate(images_dir: Path, cols: int, rows: int) -> None:
    object_points, image_points, image_size = find_corners(images_dir, cols, rows)

    if len(object_points) < 5:
        print("Not enough usable images (need at least 5). Aborting.")
        return

    error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    print(f"\nReprojection error: {error:.4f} px (lower is better, <1.0 is good)")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CALIBRATION_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
    )
    print(f"\nSaved calibration to {CALIBRATION_FILE}")


def undistort(image_path: Path) -> None:
    if not CALIBRATION_FILE.exists():
        print(f"No calibration file at {CALIBRATION_FILE}. Run calibration first.")
        return

    data = np.load(CALIBRATION_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read {image_path}")
        return

    fixed = cv2.undistort(img, camera_matrix, dist_coeffs)

    out_path = image_path.with_name(image_path.stem + "_undistorted.jpg")
    cv2.imwrite(str(out_path), fixed)
    print(f"Saved undistorted image to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
        help="Folder of checkerboard photos to calibrate from",
    )
    parser.add_argument("--cols", type=int, default=BOARD_COLS, help="Inner corners across")
    parser.add_argument("--rows", type=int, default=BOARD_ROWS, help="Inner corners down")
    parser.add_argument(
        "--undistort", type=Path, default=None,
        help="Skip calibration; undistort this image using the saved calibration",
    )
    args = parser.parse_args()

    if args.undistort:
        undistort(args.undistort)
    else:
        calibrate(args.images_dir, args.cols, args.rows)


if __name__ == "__main__":
    main()
