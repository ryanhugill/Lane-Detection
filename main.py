import numpy as np
import cv2
import time
from grab_screen import grab_screen
from lane_detection import hls_filter, lane_detection_algorithm, draw_lanes


COUNTDOWN = True

SCREEN_WIDTH = 900
SCREEN_HEIGHT = int(0.5625 * SCREEN_WIDTH)
SCREEN_POSITION = (1920 - SCREEN_WIDTH, 40, 1920, SCREEN_HEIGHT)

ROI_HEIGHT = int(0.25 * SCREEN_HEIGHT)

LANES_HLS_VALUES = {'white': [np.array([60, 240, 0]), np.array([100, 255, 255])],
                    'yellow': [np.array([80, 180, 200]), np.array([100, 200, 255])]}


def main():
    last_time = time.time()

    while True:
        frame = np.array(grab_screen(SCREEN_POSITION))
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_hls = hls_filter(frame, LANES_HLS_VALUES)
        cv2.imshow('HLS Frame', frame_hls)

        left_lane_coords, right_lane_coords = lane_detection_algorithm(frame_hls, roi_height=ROI_HEIGHT, n_windows=15)

        frame = draw_lanes(frame, [left_lane_coords, right_lane_coords], color=(0, 255, 0))
        cv2.imshow('Processed Frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        print(f'FPS: {1 / (time.time() - last_time)}')

        # cv2.putText(frame, f'FPS: {round(1 / (time.time() - last_time), 2)}', org=(30, 30),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)

        last_time = time.time()

        # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    if COUNTDOWN:
        for t in reversed(range(3)):
            print(t + 1)
            time.sleep(1)

    main()
