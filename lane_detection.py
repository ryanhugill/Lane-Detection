import numpy as np
import cv2


def hls_filter(frame, lanes_hls_values):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    masks = []

    for lane_color in lanes_hls_values:
        mask = cv2.inRange(frame, lowerb=lanes_hls_values[lane_color][0],
                           upperb=lanes_hls_values[lane_color][1])
        masks.append(mask)

    return np.bitwise_or.reduce(np.array(masks))


def lane_detection_algorithm(frame, roi_height, n_windows=20):
    height, width = frame.shape

    left_lane_coords = []
    right_lane_coords = []

    for window in range(n_windows):
        window_lower = int(((window * (height - roi_height)) / n_windows) + roi_height)
        window_upper = int((((window + 1) * (height - roi_height)) / n_windows) + roi_height)
        window_pixels = frame[window_lower: window_upper]

        column_totals = np.sum(window_pixels, axis=0)
        frame_center = int(width / 2)

        left_lane_x = np.argmax(column_totals[:frame_center])

        if left_lane_x != 0:
            left_lane_coords.append([left_lane_x, window_upper])

        right_lane_x = np.argmax(column_totals[frame_center:]) + frame_center

        if right_lane_x != frame_center:
            right_lane_coords.append([right_lane_x, window_upper])

    return left_lane_coords, right_lane_coords


def draw_lanes(frame, lanes, color=(0, 255, 0)):
    for lane in lanes:
        prev_coords = None

        for coords in lane:
            if prev_coords is not None:
                cv2.line(frame, prev_coords, coords, color, thickness=5)

            prev_coords = coords

    return frame
