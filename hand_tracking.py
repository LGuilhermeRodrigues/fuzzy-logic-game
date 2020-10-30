import cv2
import numpy as np

image_src = None
pixel = (20, 60, 80)  # some random default
upper = (20, 60, 80)
lower = (20, 60, 80)


# mouse callback function
def pick_color(event, x, y, flags, param):
    global pixel, image_src, upper, lower
    image_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y, x]
        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 20, pixel[1] + 20, pixel[2] + 80])
        lower = np.array([pixel[0] - 20, pixel[1] - 20, pixel[2] - 80])


def show_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=8)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours
    if len(cnts):
        cnt = max(cnts, key=lambda x: cv2.contourArea(x))
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        cv2.drawContours(mask, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(mask, (cX, cY), 7, (0, 255, 255), -1)
        cv2.putText(mask, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cX = 0
        cY = 0

    # show the windows
    cv2.imshow('mask', mask)
    return cX, cY


def main():
    cap = cv2.VideoCapture(0)
    global pixel, image_src, upper, lower  # so we can use it in mouse callback
    frame_number = 0
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) >= 1:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image_src = frame

        # define region of interest
        roi = frame[40:480-40, 120:640-120]
        player_x, player_y = show_mask(roi)
        cv2.putText(frame, f'Player: {player_x},{player_y}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 10, 10), 2)
        # camera resolution = 480,640
        cv2.rectangle(frame, (120, 40), (640-120, 480-40), (255, 10, 10), 0)

        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', pick_color)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        if k == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
