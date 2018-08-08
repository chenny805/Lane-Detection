# approach using cape cod video with region of interest
import cv2
import numpy as np


file = cv2.VideoCapture("motocycle.mp4")

# color range for color filtering
sensitivity = 15
lower_white = np.array([0, 0, 255-sensitivity])
upper_white = np.array([255, sensitivity, 255])



l_left = [0, 720]
l_right = [1280, 720]
t_left = [340, 100]
t_right = [940, 100]
vertices = [np.array([l_left, t_left, t_right, l_right], dtype=np.int32)]




# ROI
# Only keep the region of the image defined by the polygon
# formed from 'vertices'. The rest of the image is set to black
def region_of_interest(img, vertices):
    # defining a blank mask
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



# process video frame
while True:
    ret, video_frame = file.read()

    # loop video
    if not ret:
        file = cv2.VideoCapture()
        continue

    # use Gaussian blur to remove noise
    blur_frame = cv2.GaussianBlur(region_of_interest(video_frame, vertices), (5, 5), 0)
    # convert to HSV color space to extract pixel intensity
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    # create mask using color range
    mask = cv2.inRange(hsv, lower_white, upper_white)
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