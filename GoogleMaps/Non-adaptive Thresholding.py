import cv2
import numpy as np

# --- Config ---
input_path = "satellite7.png"
output_path = "mask.png"

img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)


lower_white_yellow = np.array([0, 0, 180])
upper_white_yellow = np.array([255, 30, 255])
color_mask_initial = cv2.inRange(hsv, lower_white_yellow, upper_white_yellow)


final_mask = cv2.bitwise_not(color_mask_initial)



cv2.imwrite(output_path, final_mask)
