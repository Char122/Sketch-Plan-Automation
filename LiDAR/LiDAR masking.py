import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import hashlib
import e57 as E57_PARSER
import open3d as o3d


E57_INPUT_FILE = "streets 1.e57" #this file is massive so you can ask Amelia for the thumb drive.
FULL_TEMP_PLY_FILE = "streets 1_temp.ply"
OUTPUT_IMAGE_PATH = 'streets 1_scalar_map.png'
OUTPUT_MASK_PATH = "streets 1_final_mask.png"

#customisable parameters
ROI_X_RANGE = [-21.8947, 31.2983]
ROI_Y_RANGE = [-14.9315, 30.3825]
ROI_Z_RANGE = [-1.55745, -0.78045]

RESOLUTION_M_PER_PIXEL = 0.03
SUBSAMPLE_RATIO = 1
PROJECTION_MODE = 'MAX'

#noise filter
APPLY_SOR_FILTER = True
SOR_NB_POINTS = 20
SOR_STD_RATIO = 2.0

#paints empty spaces
APPLY_INPAINTING = False
INPAINTING_RADIUS = 5
INPAINTING_METHOD = cv2.INPAINT_TELEA
MAX_HOLE_AREA_TO_INPAINT_PX = 100000

SCALAR_FIELD_TO_MAP = 'intensity'
SCALAR_MAX = 255.0


SCALING_METHOD = 'LOG' #'LOG' or 'LINEAR'
LOG_SCALING_CONSTANT = 0.1


COLORMAP_CONFIG = 'default_linear_grayscale'
N_COLOR_BINS = 50

#this stops code from crashing laptop by overloading memory 200m points ~ 20GB of mem
MAX_POINTS_FOR_MEMORY_SAFETY = 200_000_000


MIN_LANE_AREA_PX = 50

#adaptive thresholding
ADAPTIVE_BLOCK_SIZE = 801
ADAPTIVE_C_CONSTANT = 0.0001


def get_filtered_ply_path(base_filename, roi_x, roi_y, roi_z, scalar_field_name, scalar_max, RESOLUTION_M_PER_PIXEL,
                          MAX_POINTS_FOR_MEMORY_SAFETY, PROJECTION_MODE, SUBSAMPLE_RATIO,
                          APPLY_SOR_FILTER, SOR_NB_POINTS, SOR_STD_RATIO,
                          APPLY_INPAINTING, INPAINTING_RADIUS, INPAINTING_METHOD, MAX_HOLE_AREA_TO_INPAINT_PX,
                          SCALING_METHOD, LOG_SCALING_CONSTANT):
    config_data = (str(roi_x) + str(roi_y) + str(roi_z) + str(scalar_field_name) + str(scalar_max) + str(
        RESOLUTION_M_PER_PIXEL) + str(MAX_POINTS_FOR_MEMORY_SAFETY) + str(PROJECTION_MODE) + str(SUBSAMPLE_RATIO) +
                   str(APPLY_SOR_FILTER) + str(SOR_NB_POINTS) + str(SOR_STD_RATIO) +
                   str(APPLY_INPAINTING) + str(INPAINTING_RADIUS) + str(INPAINTING_METHOD) + str(
                MAX_HOLE_AREA_TO_INPAINT_PX) +
                   str(SCALING_METHOD) + str(LOG_SCALING_CONSTANT))
    hash_digest = hashlib.sha256(config_data.encode('utf-8')).hexdigest()[:8]
    return base_filename.replace(".e57", f"_filtered_{hash_digest}.ply")


def load_ply_data(filepath, scalar_field_to_map):
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        X = np.asarray(pcd.points)[:, 0]
        Y = np.asarray(pcd.points)[:, 1]
        Z = np.asarray(pcd.points)[:, 2]

        I_temp = np.array([])
        try:
            # Check for standard intensity field first
            I_temp = np.asarray(pcd.get_intensity())
        except AttributeError:
            pass

        if I_temp.size > 0 and np.max(I_temp) > 0:
            I = I_temp.astype(np.float32)
        elif len(pcd.colors) > 0:
            I_normalized = np.asarray(pcd.colors)[:, 0]
            if scalar_field_to_map == 'z_height':
                I = Z.copy()
            else:
                I = I_normalized * SCALAR_MAX
        else:
            I = Z.copy()


        data = np.stack([X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32), I.astype(np.float32)],
                        axis=1)
        return data

    except Exception as e:
        print(f"CRITICAL ERROR: Open3D failed to read PLY file '{filepath}'. (Error: {e})")
        return None


def save_filtered_ply(data, filepath):
    try:
        pcd_to_write = o3d.geometry.PointCloud()
        pcd_to_write.points = o3d.utility.Vector3dVector(data[:, 0:3])
        scalar_data = data[:, 3].astype(np.float64)

        try:
            pcd_to_write.set_intensity(o3d.utility.DoubleVector(scalar_data))
        except AttributeError:
            max_val = np.max(scalar_data) if np.max(scalar_data) > 0 else 1.0
            I_normalized = np.clip(scalar_data / max_val, 0.0, 1.0)
            colors = np.stack([I_normalized, I_normalized, I_normalized], axis=1)
            pcd_to_write.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(filepath, pcd_to_write, write_ascii=True)
        print(f"Saved point cloud to cache: {filepath}")
        return True
    except Exception as e:
        print(f"WARNING: Failed to save PLY cache file. (Error: {e})")
        return False



def load_point_cloud(filename, full_temp_ply_path):
    if os.path.exists(full_temp_ply_path):
        print(f"\n--- MEDIUM PATH: Loading Generic PLY Cache ({full_temp_ply_path}) ---")
        return load_ply_data(full_temp_ply_path, SCALAR_FIELD_TO_MAP)

    try:
        fields = ['cartesianX', 'cartesianY', 'cartesianZ', 'intensity', 'reflectivity']

        if hasattr(E57_PARSER, 'E57File'):
            e57_file = E57_PARSER.E57File(filename)
            scan = e57_file.read_scan(0, fields=fields)
        else:
            func = getattr(E57_PARSER, 'read_points', None) or getattr(E57_PARSER, 'read')
            dataframes = func(filename)
            scan = dataframes[0] if isinstance(dataframes, (list, tuple)) and len(dataframes) > 0 else dataframes

        if scan is None: raise ValueError("Failed to extract scan data.")

        # Coordinate Extraction
        X, Y, Z = None, None, None
        points_data = getattr(scan, 'points', None)
        if points_data is not None and isinstance(points_data, np.ndarray) and points_data.ndim == 2 and \
                points_data.shape[1] >= 3:
            X, Y, Z = points_data[:, 0], points_data[:, 1], points_data[:, 2]
        else:
            X = getattr(scan, 'cartesianX', getattr(scan, 'x', None))
            Y = getattr(scan, 'cartesianY', getattr(scan, 'y', None))
            Z = getattr(scan, 'cartesianZ', getattr(scan, 'z', None))

            if X is not None and not isinstance(X, np.ndarray): X = np.array(X)
            if Y is not None and not isinstance(Y, np.ndarray): Y = np.array(Y)
            if Z is not None and not isinstance(Z, np.ndarray): Z = np.array(Z)

        if X is None or Y is None or Z is None or not (len(X) == len(Y) == len(Z)):
            raise ValueError("Coordinate arrays (X, Y, Z) are invalid or mismatched lengths.")

        total_points = len(X)
        if total_points == 0:
            print("WARNING: Point cloud contains zero points. Aborting conversion.")
            return None

        if total_points > MAX_POINTS_FOR_MEMORY_SAFETY:
            X = X[:MAX_POINTS_FOR_MEMORY_SAFETY]
            Y = Y[:MAX_POINTS_FOR_MEMORY_SAFETY]
            Z = Z[:MAX_POINTS_FOR_MEMORY_SAFETY]
            total_points = len(X)

        I = None
        fields_to_check = ['intensity', 'reflectivity', 'z_height']

        for field in fields_to_check:
            I_temp = Z.copy() if field == 'z_height' else getattr(scan, field, None)
            if I_temp is not None:
                I_temp = np.array(I_temp).squeeze()
                if len(I_temp) > total_points: I_temp = I_temp[:total_points]
                if I_temp.ndim == 1 and len(I_temp) == total_points and np.max(I_temp) > 0:
                    I = I_temp.flatten().astype(np.float32)
                    break

        if I is None:
            print("WARNING: No valid scalar fields found. Using Z-Height for Generic PLY Cache.")
            I = Z.flatten().astype(np.float32)

        # STACK and SAVE TIER 2 CACHE
        raw_data_array = np.stack([X.flatten().astype(np.float32),
                                   Y.flatten().astype(np.float32),
                                   Z.flatten().astype(np.float32),
                                   I], axis=1)

        save_filtered_ply(raw_data_array, full_temp_ply_path)
        return load_ply_data(full_temp_ply_path, SCALAR_FIELD_TO_MAP)

    except Exception as e:
        print(f"CRITICAL ERROR: E57 parsing failed (Error: {e}).")
        return None

def subsample_point_cloud(data, ratio):
    if data is None or ratio >= 1.0:
        return data

    total_points = len(data)
    sample_size = int(total_points * ratio)

    if sample_size <= 0 and total_points > 0: sample_size = 1
    if sample_size >= total_points: return data

    indices = np.random.choice(total_points, sample_size, replace=False)
    subsampled_data = data[indices]

    print(f"Uniformly subsampled {total_points} points down to {sample_size} ({ratio * 100:.1f}%)")
    return subsampled_data


def filter_point_cloud(data):
    if data is None: return None
    print(f"Points loaded: {len(data)}")

    mask_x = (data[:, 0] >= ROI_X_RANGE[0]) & (data[:, 0] <= ROI_X_RANGE[1])
    mask_y = (data[:, 1] >= ROI_Y_RANGE[0]) & (data[:, 1] <= ROI_Y_RANGE[1])
    mask_z = (data[:, 2] >= ROI_Z_RANGE[0]) & (data[:, 2] <= ROI_Z_RANGE[1])

    mask = mask_x & mask_y & mask_z

    filtered_data = data[mask]
    print(f"Points after Cropping (ROI filter): {len(filtered_data)}")
    return filtered_data


def apply_statistical_outlier_removal(data, nb_points, std_ratio):
    if data is None or len(data) == 0:
        return data

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])

    scalar_data = data[:, 3].copy()

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_points, std_ratio=std_ratio)

    filtered_points = np.asarray(pcd.points)[ind]
    filtered_scalar = scalar_data[ind]

    filtered_data = np.hstack((filtered_points, filtered_scalar[:, np.newaxis]))

    print(
        f"SOR Filter Applied: {len(data)} points reduced to {len(filtered_data)} (removed {len(data) - len(filtered_data)} outliers).")
    return filtered_data


def project_to_2d_intensity_map(data):

    if data is None or len(data) == 0:
        return np.zeros((100, 100), dtype=np.float32)

    x_range_m = ROI_X_RANGE[1] - ROI_X_RANGE[0]
    y_range_m = ROI_Y_RANGE[1] - ROI_Y_RANGE[0]

    width_px = int(np.ceil(x_range_m / RESOLUTION_M_PER_PIXEL))
    height_px = int(np.ceil(y_range_m / RESOLUTION_M_PER_PIXEL))

    scalar_map = np.zeros((height_px, width_px), dtype=np.float32)

    X = data[:, 0]
    Y = data[:, 1]
    S = data[:, 3]

    u_coords = ((X - ROI_X_RANGE[0]) / RESOLUTION_M_PER_PIXEL).astype(np.int32)
    v_coords = ((Y - ROI_Y_RANGE[0]) / RESOLUTION_M_PER_PIXEL).astype(np.int32)

    valid_mask = (u_coords >= 0) & (u_coords < width_px) & \
                 (v_coords >= 0) & (v_coords < height_px)

    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    S_valid = S[valid_mask]

    v_coords_flipped = height_px - 1 - v_coords

    print("Projection Mode: MAX (Highest intensity point dominates)")
    for u, v, scalar in zip(u_coords, v_coords_flipped, S_valid):
        scalar_map[v, u] = max(scalar_map[v, u], scalar)

    print(f"Projected map size: {width_px}x{height_px} pixels")
    return scalar_map


def apply_inpainting(scalar_map, radius, method, max_shadow_area):

    if not APPLY_INPAINTING:
        return scalar_map, np.zeros_like(scalar_map, dtype=np.uint8)

    print(f"Applying Inpainting (Radius: {radius}, Max Area: {max_shadow_area}px)...")

    shadow_mask = (scalar_map == 0).astype(np.uint8) * 255


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(shadow_mask, 8, cv2.CV_32S)


    small_shadows_mask = np.zeros_like(shadow_mask, dtype=np.uint8)
    large_shadows_mask = np.zeros_like(shadow_mask, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        component_mask = (labels == i).astype(np.uint8) * 255

        if area <= max_shadow_area:
            small_shadows_mask = cv2.bitwise_or(small_shadows_mask, component_mask)
        else:
            large_shadows_mask = cv2.bitwise_or(large_shadows_mask, component_mask)

    final_result = scalar_map.copy()


    if np.sum(small_shadows_mask) > 0:

        non_zero_mask = scalar_map > 0
        non_zero_max = np.max(scalar_map[non_zero_mask]) if np.any(non_zero_mask) else 1.0
        scaled_map_8u = ((np.clip(scalar_map, 0, non_zero_max) / non_zero_max) * 255).astype(np.uint8)
        inpainted_8u = cv2.inpaint(scaled_map_8u, small_shadows_mask, radius, method)
        inpainted_map_float = (inpainted_8u.astype(np.float32) / 255) * non_zero_max

        final_result[small_shadows_mask > 0] = inpainted_map_float[small_shadows_mask > 0]
        print(f"Filled {np.sum(small_shadows_mask) // 255} small shadow pixels via inpainting.")
    else:
        print("No small shadows found for inpainting.")

    if np.sum(large_shadows_mask) > 0:
        final_result[large_shadows_mask > 0] = SCALAR_MAX
        print(f"Filled {np.sum(large_shadows_mask) // 255} large shadow pixels with white ({SCALAR_MAX}).")

    return final_result, large_shadows_mask


def convert_to_8bit_grayscale(scalar_map, method='LINEAR', log_constant=1.0):
    if scalar_map.size == 0:
        return np.zeros((100, 100), dtype=np.uint8)

    # 1. Clip and calculate non-zero max value (used for normalization)
    clipped_map = np.clip(scalar_map, 0, SCALAR_MAX)
    I_max = np.max(clipped_map) if np.max(clipped_map) > 0 else 1.0

    if method == 'LINEAR':
        print("Scaling Method: LINEAR")
        # Scale the map linearly from [0, I_max] to [0.0, 1.0]
        normalized_map = clipped_map / I_max

    elif method == 'LOG':
        print(f"Scaling Method: LOG (Constant: {log_constant})")
        # Apply Logarithmic scaling: log(I + C)

        # Calculate the log of the max value for normalization
        log_max = np.log(I_max + log_constant)

        if log_max <= 0:  # Safety check
            log_max = 1.0

        # Apply log to the entire map, then normalize
        log_map = np.log(clipped_map + log_constant)
        normalized_map = log_map / log_max

        # Note: Any true zero pixels (e.g., in the original shadow regions,
        # which are not covered by the current inpainted/filled areas) will be near zero after log scaling.

    else:
        print(f"WARNING: Unknown scaling method '{method}'. Falling back to LINEAR.")
        normalized_map = clipped_map / I_max

    # 2. Scale to 0-255 and convert to 8-bit unsigned integer (uint8)
    gray_map_8u = (normalized_map * 255).astype(np.uint8)

    return gray_map_8u


def clean_lane_mask(mask, min_area_px):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    final_mask = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            component_mask = (labels == i).astype(np.uint8) * 255
            final_mask = cv2.bitwise_or(final_mask, component_mask)

    # Morphological closing and opening for smoothing
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_small)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_small)

    return final_mask


if __name__ == '__main__':

    #check for cache pathway
    filtered_ply_path = get_filtered_ply_path(E57_INPUT_FILE, ROI_X_RANGE, ROI_Y_RANGE, ROI_Z_RANGE,
                                              SCALAR_FIELD_TO_MAP, SCALAR_MAX, RESOLUTION_M_PER_PIXEL,
                                              MAX_POINTS_FOR_MEMORY_SAFETY, PROJECTION_MODE, SUBSAMPLE_RATIO,
                                              APPLY_SOR_FILTER, SOR_NB_POINTS, SOR_STD_RATIO,
                                              APPLY_INPAINTING, INPAINTING_RADIUS, INPAINTING_METHOD,
                                              MAX_HOLE_AREA_TO_INPAINT_PX,
                                              SCALING_METHOD, LOG_SCALING_CONSTANT)

    cropped_points = None

    # Check for TIER 1
    if os.path.exists(filtered_ply_path):
        print(f"\n--- TIER 1: FASTEST PATH: Loading Hashed Cache ({filtered_ply_path}) ---")
        cropped_points = load_ply_data(filtered_ply_path, SCALAR_FIELD_TO_MAP)
        if cropped_points is not None and len(cropped_points) > 0:
            print(f"Loaded {len(cropped_points)} points from hash cache.")
        else:
            cropped_points = None

    # Proceed to TIER 2/3 if no Tier 1
    if cropped_points is None:
        raw_points = load_point_cloud(E57_INPUT_FILE, FULL_TEMP_PLY_FILE)

        if raw_points is None:
            print("Processing aborted due to loading error.")
            sys.exit(1)

        raw_points = filter_point_cloud(raw_points)

        if len(raw_points) == 0:
            print("No points remaining after cropping. Check ROI parameters.")
            sys.exit(1)

        cropped_points = subsample_point_cloud(raw_points, SUBSAMPLE_RATIO)

        if len(cropped_points) == 0 and SUBSAMPLE_RATIO > 0:
            print(f"WARNING: Subsampling with ratio {SUBSAMPLE_RATIO} resulted in zero points after ROI filter.")
            sys.exit(1)

        # Apply 3D Noise Filter (SOR)
        if APPLY_SOR_FILTER:
            cropped_points = apply_statistical_outlier_removal(cropped_points, SOR_NB_POINTS, SOR_STD_RATIO)
            if len(cropped_points) == 0:
                print("No points remaining after SOR filtering. Check SOR parameters.")
                sys.exit(1)

        # Save the points to the hashed cache
        save_filtered_ply(cropped_points, filtered_ply_path)

    # --- Final Processing and Visualization ---
    if cropped_points is not None and len(cropped_points) > 0:

        scalar_map_2d = project_to_2d_intensity_map(cropped_points)

        scalar_map_2d_inpainted, large_shadows_mask = apply_inpainting(
            scalar_map_2d,
            INPAINTING_RADIUS,
            INPAINTING_METHOD,
            MAX_HOLE_AREA_TO_INPAINT_PX
        )

        # 2. Convert to 8-bit Grayscale Map (Log Scale)
        final_grayscale_image = convert_to_8bit_grayscale(
            scalar_map_2d_inpainted,
            method=SCALING_METHOD,
            log_constant=LOG_SCALING_CONSTANT
        )

        cv2.imwrite(OUTPUT_IMAGE_PATH, final_grayscale_image)
        print(f"\nSaved 2D Grayscale Scalar Map (Log Scaled) to: {OUTPUT_IMAGE_PATH}")

        gray_img = final_grayscale_image.copy()

        gray_blur = cv2.GaussianBlur(gray_img, (1, 1), 0)

        adaptive_mask = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C_CONSTANT
        )

        initial_adaptive_mask = cv2.bitwise_not(adaptive_mask)


        if large_shadows_mask.size > 0:
            initial_adaptive_mask[large_shadows_mask > 0] = 0
            print("Forcing large shadow areas to black in the adaptive mask to exclude them from the result.")


        final_cleaned_mask = clean_lane_mask(initial_adaptive_mask, MIN_LANE_AREA_PX)


        cv2.imwrite(OUTPUT_MASK_PATH, final_cleaned_mask)
        print(f"Saved Final Binary Mask to: {OUTPUT_MASK_PATH}")


        final_output_highlight = cv2.cvtColor(final_grayscale_image, cv2.COLOR_GRAY2RGB)
        final_output_highlight[final_cleaned_mask > 0] = [255, 0, 255]

        plt.figure(figsize=(18, 9))

        plt.subplot(1, 3, 1)
        original_grayscale_image_log = convert_to_8bit_grayscale(
            scalar_map_2d,
            method=SCALING_METHOD,
            log_constant=LOG_SCALING_CONSTANT
        )
        plt.imshow(original_grayscale_image_log, cmap='gray')
        plt.title(
            f'1. Log-Scaled Projection (w/ black shadows)')

        plt.subplot(1, 3, 2)
        plt.imshow(final_grayscale_image, cmap='gray')
        plt.title(f'2. Log-Scaled & Inpainted Map (Large Shadows White)')

        plt.subplot(1, 3, 3)
        plt.imshow(final_cleaned_mask, cmap='gray')
        plt.title(f'3. Final Cleaned Mask (Post-Correction)')

        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(9, 9))
        plt.imshow(final_output_highlight)
        plt.title('Final Cleaned Lane Mask Overlay (Magenta on Log-Scaled Background)')
        plt.show()

    else:
        print("Processing finished: No points were loaded or remaining after filtering.")
