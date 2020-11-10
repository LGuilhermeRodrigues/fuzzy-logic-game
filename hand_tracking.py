import sys

import cv2
import numpy as np
import random
import math

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


def draw_game(frame, start_x, start_y, num_pixels, player_pos):
    player_x, player_y = player_pos
    border = 40
    interval_x = start_x + border, start_x + num_pixels - border
    interval_y = start_y + border, start_y + num_pixels - border
    game_x = random.randint(interval_x[0], interval_x[1])
    game_y = random.randint(interval_y[0], interval_y[1])

    while computer_distance(player_x, player_y, game_x, game_y) < 200:
        game_x = random.randint(interval_x[0], interval_x[1])
        game_y = random.randint(interval_y[0], interval_y[1])

    print('teste ', game_x, ',', game_y)

    # cv2.rectangle(frame, (start_x+border, start_y + border), (start_x+num_pixels-border, start_y + num_pixels - border), (255, 10, 10), 0)
    return game_x, game_y


def computer_distance(player_x, player_y, coin_x, coin_y):
    distance = math.sqrt(((player_x - coin_x) ** 2) + ((player_y - coin_y) ** 2))
    return distance


def main():
    cap = cv2.VideoCapture(0)
    global pixel, image_src, upper, lower  # so we can use it in mouse callback
    frame_number = 0
    seconds = 0
    coin_left = 11
    points = 0
    start_button = (int(640 / 2) - 20, int(480 / 2) - 40)
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    coin_pos = (None, None)
    while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) >= 1:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image_src = frame

        # define region of interest
        roi = frame[40:480 - 40, 120:640 - 120]
        player_x, player_y = show_mask(roi)
        player_x += 120
        player_y += 40
        cv2.putText(frame, f'Player: {player_x},{player_y}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 10, 10), 2)
        # camera resolution = 480,640
        cv2.rectangle(frame, (120, 40), (640 - 120, 480 - 40), (255, 10, 10), 0)
        if pixel[0] == 20 and pixel[1] == 60 and pixel[2] == 80:
            cv2.putText(frame, 'Choose the marker', (int(640 / 2) - 100, int(480 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        else:
            if start_button is not None:
                cv2.putText(frame, 'Move the marker here to start', (int(640 / 2) - 120, int(480 / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 100), 2)
                cv2.circle(frame, start_button, 20, (0, 255, 50), -1)
                start_distance = computer_distance(player_x, player_y, start_button[0], start_button[1])
                if start_distance < 30:
                    start_button = None
                    start_position = (player_x, player_y)
                    points = 0
                    seconds = 0
                    coin_left = 10
            else:
                # game has started
                player_position = (player_x, player_y)

        coin_found = False
        if not coin_pos[0]:
            coin_x, coin_y = draw_game(frame, 120, 40, 400, (player_x, player_y))
            coin_pos = (coin_x, coin_y)
            coin_found = True
        cv2.putText(frame, f'Coin: {coin_pos[0]},{coin_pos[1]}', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 10, 10), 2)
        cv2.circle(frame, (coin_pos[0], coin_pos[1]), 20, (255, 10, 10), -1)
        cv2.putText(frame, '25', (coin_pos[0] - 12, coin_pos[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        distance = computer_distance(player_x, player_y, coin_pos[0], coin_pos[1])
        cv2.putText(frame, f'Distance: {distance:.0f}', (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 10, 10), 2)

        if distance < 30:
            coin_pos = (None, None)

        frame_number += 1
        if frame_number % 3 == 0:
            seconds += 0.1

        cv2.putText(frame, f'Seconds: {seconds * 1.0:.1f}', (400, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 10, 10), 2)

        if coin_found:
            coin_left -= 1
        if coin_left == 0:
            points_final = points
        if coin_left <= 0:
            cv2.putText(frame, f'Points: {points_final}', (int(640 / 2) - 30, int(480 / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        cv2.putText(frame, f'Coins Left: {coin_left}', (230, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 10, 10), 2)
        cv2.putText(frame, f'Points: {points}', (90, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 10, 10), 2)

        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', pick_color)

        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break
        if k == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
    sys.exit()
