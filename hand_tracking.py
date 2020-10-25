import cv2
import imutils as imutils
import numpy as np
import math

image_hsv = None   # global ;(
image_src = None
pixel = (20,60,80) # some stupid default
upper = None
lower = None

# mouse callback function
def pick_color(event,x,y,flags,param):
    global image_hsv, pixel, image_src, upper, lower
    image_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 20, pixel[1] + 20, pixel[2] + 80])
        lower =  np.array([pixel[0] - 20, pixel[1] - 20, pixel[2] - 80])
        print(pixel, lower, upper)

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)


def main():
    cap = cv2.VideoCapture(0)
    # COLOR PICKER
    global image_hsv, pixel, image_src, upper,lower # so we can use it in mouse callback

    ret, image_src = cap.read()
    cv2.imshow("bgr", image_src)
    cv2.setMouseCallback('bgr', pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kernel = np.ones((3, 3), np.uint8)

    frame_number = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # define region of interest
        roi = frame[100:300, 100:300]

        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.dilate(mask, kernel, iterations=8)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(contours)
        cnts = contours
        if len(cnts):
            cnt = max(cnts, key=lambda x: cv2.contourArea(x))
            """epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            hull = cv2.convexHull(cnt)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            arearatio = ((areahull - areacnt) / areacnt) * 100
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
            l = 0
            # cv2.drawContours(frame, hull, 0, (255, 0, 0), 1, 8)"""
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.drawContours(mask, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(mask, (cX, cY), 7, (0, 255, 255), -1)
            cv2.putText(mask, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        frame_number += 1
        cv2.putText(frame, str(frame_number), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__=='__main__':
    main()
