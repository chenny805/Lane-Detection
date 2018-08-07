# approach using cape cod video with region of interest
import cv2
import numpy as np


file = cv2.VideoCapture("lane_test.mp4")

# color range for color filtering
low_yellow = np.array([18, 94, 140])
upper_yellow = np.array([48, 255, 255])

# process video frame
while True:
    ret, video_frame = file.read()

    # loop video
    if not ret:
        file = cv2.VideoCapture()
        continue

    # use Gaussian blur to remove noise
    blur_frame = cv2.GaussianBlur(video_frame, (5, 5), 0)
    # convert to HSV color space to extract pixel intensity
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    # create mask using color range
    mask = cv2.inRange(hsv, low_yellow, upper_yellow)
    # canny edge detection to find the edges of the lanes
    canny_edge = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(canny_edge, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2 , y2 = line[0]
            cv2.line(video_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    # display window
    cv2.imshow("window", video_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
file.release()