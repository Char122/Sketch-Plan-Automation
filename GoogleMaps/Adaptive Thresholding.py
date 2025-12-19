import cv2
import numpy as np

input_path = "satellite8.png"
output_path = "maskcodenew1.png"
third_output_path = "maskcodenew2.png"

MIN_WIDTH_PX = 1
MAX_WIDTH_PX = 15

MIN_LANE_AREA = 1
MIN_LANE_LINE_RATIO = 1

MAX_LANE_AREA = 10000
MAX_LANE_SQUARE_RATIO = 1

GRADIENT_THRESHOLD = 30


img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]


gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
gray_blur = cv2.GaussianBlur(gray, (1, 1), 0)
road_mask = np.ones_like(gray, dtype=np.uint8) * 255


grad_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
mask = np.zeros_like(gray, dtype=np.uint8)


for y in range(h):
    gx = grad_x[y]
    for x in range(w - MAX_WIDTH_PX):
        if gx[x] > GRADIENT_THRESHOLD:
            for d in range(MIN_WIDTH_PX, MAX_WIDTH_PX + 1):
                if gx[x + d] < -GRADIENT_THRESHOLD:
                    mask[y, x] = 255
                    mask[y, x + d] = 255
                    break

for x in range(w):
    gy = grad_y[:, x]
    for y in range(h - MAX_WIDTH_PX):
        if gy[y] > GRADIENT_THRESHOLD:
            for d in range(MIN_WIDTH_PX, MAX_WIDTH_PX + 1):
                if gy[y + d] < -GRADIENT_THRESHOLD:
                    mask[y, x] = 255
                    mask[y + d, x] = 255
                    break


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)



lane_contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_lane_mask = np.zeros_like(gray, dtype=np.uint8) # Mask to hold only valid lanes

for c in lane_contours:
    area = cv2.contourArea(c)

    if area < MIN_LANE_AREA or area > MAX_LANE_AREA:
        continue

    x, y, w_c, h_c = cv2.boundingRect(c)

    if h_c == 0 or w_c == 0: continue

    aspect_ratio_len_div_wid = max(w_c, h_c) / min(w_c, h_c)
    aspect_ratio_squareness = min(w_c, h_c) / max(w_c, h_c)


    if aspect_ratio_len_div_wid >= MIN_LANE_LINE_RATIO and aspect_ratio_squareness <= MAX_LANE_SQUARE_RATIO:
        cv2.drawContours(filtered_lane_mask, [c], -1, 255, thickness=cv2.FILLED)


combined_mask = filtered_lane_mask



car_candidates = cv2.inRange(gray, 0, 80)
car_candidates = cv2.bitwise_and(car_candidates, road_mask)


car_subtraction_mask = np.zeros_like(gray, dtype=np.uint8)

contours, _ = cv2.findContours(car_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    area = cv2.contourArea(c)

    if 300 < area < 15000:
        x, y, w_c, h_c = cv2.boundingRect(c)
        aspect_ratio = w_c / h_c

        if 0.5 < aspect_ratio < 2.5:
            cv2.drawContours(car_subtraction_mask, [c], -1, 255, thickness=cv2.FILLED)


car_subtraction_mask_inv = cv2.bitwise_not(car_subtraction_mask)
final_mask_no_cars = cv2.bitwise_and(combined_mask, combined_mask, mask=car_subtraction_mask_inv)


final_mask = cv2.bitwise_not(final_mask_no_cars)
cv2.imwrite(output_path, final_mask)
cv2.imwrite(third_output_path, car_subtraction_mask)

cv2.imshow("Mask", final_mask)

cv2.waitKey(0) # Wait indefinitely until a key is pressed
cv2.destroyAllWindows() # Close all OpenCV windows
