import cv2

clicks = []  # stores (x, y) for every click

def get_regions():
    img = "/home/kkristjansson/DTU/CAPeX/Computer-Vision-CapEx-/synthesis-monitor/capture/second_iteration/crucibles/20260828_164353_0000_undistorted.jpg"
    img = cv2.imread(img)
    clone = img.copy()


    def show_coords(event, x, y, flags, param):
        global clicks

        if event == cv2.EVENT_MOUSEMOVE:
            temp = clone.copy()
            text = f"({x}, {y})"
            cv2.putText(temp, text, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("image", temp)

        elif event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            print(f"Clicked: ({x}, {y})  |  total clicks: {len(clicks)}")

            # draw a permanent marker on the base image so it persists
            cv2.circle(clone, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("image", clone)

    # Resizable window so you can drag it bigger, or set a fixed large size below
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1200, 800)  # adjust to whatever fits your screen

    cv2.imshow("image", img)
    cv2.setMouseCallback("image", show_coords)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc to quit
            break
        elif key == ord('s'):  # optional: save clicks to file
            with open("clicks.txt", "w") as f:
                for pt in clicks:
                    f.write(f"{pt[0]}, {pt[1]}\n")
            print("Saved clicks to clicks.txt")

    cv2.destroyAllWindows()
    print("All clicks:", clicks)