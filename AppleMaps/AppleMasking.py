import cv2
import numpy as np

image_path = "applemaps.jpg"
image = cv2.imread(image_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_white_markings = np.array([0, 0, 170], dtype=np.uint8)
upper_white_markings = np.array([255, 30, 255], dtype=np.uint8)
white_markings_mask = cv2.inRange(hsv, lower_white_markings, upper_white_markings)



road_mask_inverted = cv2.bitwise_not(white_markings_mask)

cv2.imshow("Inverted Road and Markings", road_mask_inverted)
cv2.imwrite("mask.png",road_mask_inverted)

cv2.waitKey(0)
cv2.destroyAllWindows()
