import requests
import json
import os
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image



TARGET_LAT, TARGET_LON = 1.316920, 103.892253


LAT_SHIFT = 0.0
LON_SHIFT = 0.0

RADIUS_METERS = 50
PRE_CHECK_RADIUS_METERS = 200
KERB_PRE_CHECK_RADIUS_METERS = 1000


LANE_DATASET_ID = "d_fa71cc0c433275f4d2b0133358cc4fbf"
KERB_DATASET_ID = "d_45d62178079823aea49a2a7d3d7803c7"
ARROW_DATASET_ID = "d_2771fb949ceea676dd3eecade3830c5b"
WORD_DATASET_ID = "d_dfd0b6e8d5e82643d8278690e576a523"

BASE_URL = "https://api-open.data.gov.sg/v1/public/api/datasets/"
DOWNLOAD_URL_ENDPOINT = f"{BASE_URL}{{dataset_id}}/poll-download"

CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

OUTPUT_PNG_FILE = "road_features_plot.png"
LEGEND_OUTPUT_FILE = "road_features_legend.png"

ARROW_IMAGE_SIZE_M = 3.0
ARROW_IMAGE_DIR = "custom_arrow_images"
ARROW_IMAGES = {
    "A": "Right_Turn_Single.png",
    "B": "Straight_Left_Turn.png",
    "C": "Straight_Right_Turn.png",
    "D": "Straight_Single.png",
    "E": "Left_Turn_Single.png",
    "F": "Left_Right_Turn.png",
    "G": "Left_Conversional.png",
    "H": "Right_Conversional.png",
    "I": "Straight_Left_Right_Turn.png",
    "J": "Yellow_Straight_Right_Turn.png",
    "K": "Straight_Right_Turn.png",
    "L": "Right_Turn_Single.png",
    "M": "Left_Turn_Single.png",
    "N": "Straight_Single.png",
    "O": "Left_Right_Turn.png",
    "P": "Straight_Left_Turn.png",
    "Q": "U_Turn_Arrow.png",
    "DEFAULT": "Unknown_Arrow.png"
}
IMAGE_CACHE = {}

# --- 2. Style Definitions ---
DOTTED_SEP_TUPLE = (0, (1.0, 5.0))
DASHED_SEP_TUPLE = (0, (12.0, 12.0))
METERS_PER_DEGREE_LAT = 111111

LANE_STYLES = {
    "A": {"color": "black", "linewidth": 1.2, "linestyle": (0, (5.0, 5.0)), "zorder": 2},
    "A1": {"color": "black", "linewidth": 1.2, "linestyle": (0, (5.0, 5.0)), "zorder": 3},
    "A2": {"color": "black", "linewidth": 2.4, "linestyle": (0, (5.0, 5.0)), "zorder": 3},
    "A3": {"color": "yellow", "linewidth": 3.6, "linestyle": (0, (5.0, 5.0)), "zorder": 4},
    "A4": {"color": "black", "linewidth": 2.4, "linestyle": (0, (1.0, 1.0)), "zorder": 1},
    "A5": {"color": "black", "linewidth": 1.2, "linestyle": (0, (5.0, 15.0)), "zorder": 2},
    "A6": {"color": "red", "linewidth": 3.0, "linestyle": (0, (5.0, 5.0)), "zorder": 5},
    "A7": {"color": "red", "linewidth": 4.0, "linestyle": (0, (5.0, 5.0)), "zorder": 4},
    "A8": {"color": "black", "linewidth": 4.8, "linestyle": (0, (1.0, 1.0)), "zorder": 1},
    "B": {"color": "black", "linewidth": 1.2, "linestyle": (0, (10.0, 20.0)), "zorder": 1},
    "B1": {"color": "black", "linewidth": 1.2, "linestyle": (0, (10.0, 50.0)), "zorder": 3},
    "B2": {"color": "black", "linewidth": 3.0, "linestyle": (0, (10.0, 20.0)), "zorder": 4},
    "C": {"color": "black", "linewidth": 1.2, "linestyle": (0, (10.0, 5.0)), "zorder": 2},
    "C1": {"color": "black", "linewidth": 2.4, "linestyle": (0, (20.0, 10.0)), "zorder": 3},
    "D": {"color": "white", "linewidth": 1.2, "linestyle": (0, (5.0, 5.0)), "zorder": 1},
    "D1": {"color": "black", "linewidth": 1.2, "linestyle": (0, (2.5, 2.5)), "zorder": 3},
    "J": {"color": "black", "linewidth": 2.6, "linestyle": "solid", "zorder": 4},
    "R": {"color": "black", "linewidth": 3.0, "linestyle": DASHED_SEP_TUPLE, "zorder": 3},
    "N": {"color": "#eccb91", "linewidth": 2.0, "linestyle": "solid", "zorder": 5},
    "E": {"color": "black", "linewidth": 1.8, "linestyle": (0, (13.75, 13.75)), "zorder": 4},
    "F": {"color": "black", "linewidth": 1.8, "linestyle": "solid", "zorder": 5},
    "G": {"color": "black", "linewidth": 1.8, "linestyle": "solid", "zorder": 5},
    "H": {"color": "black", "linewidth": 1.2, "linestyle": "solid", "zorder": 6},
    "I": {"color": "#eccb91", "linewidth": 2, "linestyle": "solid", "zorder": 6},
    "K": {"color": "black", "linewidth": 1.2, "linestyle": "solid", "zorder": 4},
    "L": {"color": "#eccb91", "linewidth": 3.6, "linestyle": "solid", "zorder": 6},
    "M": {"color": "black", "linewidth": 2.4, "linestyle": "solid", "zorder": 3},
    "O": {"color": "black", "linewidth": 1.2, "linestyle": DOTTED_SEP_TUPLE, "zorder": 2},
    "P": {"color": "black", "linewidth": 1.2, "linestyle": DOTTED_SEP_TUPLE, "zorder": 3},
    "Q": {"color": "red", "linewidth": 5.4, "linestyle": "solid", "zorder": 7},
    "Q1": {"color": "red", "linewidth": 3.5, "linestyle": "solid", "zorder": 6},
    "Q2": {"color": "red", "linewidth": 3.0, "linestyle": DOTTED_SEP_TUPLE, "zorder": 5},
    "S": {"color": "black", "linewidth": 3.0, "linestyle": DOTTED_SEP_TUPLE, "zorder": 4},
    "T": {"color": "black", "linewidth": 3.0, "linestyle": DASHED_SEP_TUPLE, "zorder": 5},
    "U": {"color": "black", "linewidth": 1.0, "linestyle": DASHED_SEP_TUPLE, "zorder": 2},
    "V": {"color": "black", "linewidth": 4.0, "linestyle": "-.", "zorder": 7},
    "W": {"color": "black", "linewidth": 4.0, "linestyle": "solid", "zorder": 7},
    "X": {"color": "black", "linewidth": 3.0, "linestyle": DOTTED_SEP_TUPLE, "zorder": 5},
    "Y": {"color": "black", "linewidth": 1.5, "linestyle": "solid", "zorder": 3},
    "Z": {"color": "black", "linewidth": 2.0, "linestyle": DOTTED_SEP_TUPLE, "zorder": 4},
    "DEFAULT": {"color": "gray", "linewidth": 0.5, "linestyle": "-.", "zorder": 0}
}

KERB_STYLES = {
    "Generic Kerb": {"color": "brown", "linewidth": 2.5, "linestyle": "solid", "zorder": 8},
    "DEFAULT": {"color": "brown", "linewidth": 1.0, "linestyle": "solid", "zorder": 6}
}

ARROW_STYLES = {
    "A": {"typ_name": "Right Turn (Single)", "filename_key": "A"},
    "D": {"typ_name": "Straight (Single)", "filename_key": "D"},
    "E": {"typ_name": "Left Turn (Single)", "filename_key": "E"},
    "G": {"typ_name": "Left Conversional (Conv)", "filename_key": "G"},
    "H": {"typ_name": "Right Conversional (Conv)", "filename_key": "H"},
    "L": {"typ_name": "Part Time Right Turn (Single)", "filename_key": "A"},
    "M": {"typ_name": "Part Time Left Turn (Single)", "filename_key": "E"},
    "N": {"typ_name": "Part Time Straight (Single)", "filename_key": "D"},
    "B": {"typ_name": "St/Lt Turn (Combined)", "filename_key": "B"},
    "C": {"typ_name": "St/Rt Turn (Combined)", "filename_key": "C"},
    "F": {"typ_name": "Lt/Rt Turn (Combined)", "filename_key": "F"},
    "I": {"typ_name": "St/Lt/Rt Turn (Combined)", "filename_key": "I"},
    "J": {"typ_name": "St/Rt Turn-Yellow Arrow (Combined)", "filename_key": "C"},
    "K": {"typ_name": "Part-Time St/Rt Turn Shared (Combined)", "filename_key": "C"},
    "O": {"typ_name": "Part-Time Lt/Rt Turn Shared (Combined)", "filename_key": "F"},
    "P": {"typ_name": "Part Time St/Lt Turn Shared (Combined)", "filename_key": "B"},
    "Q": {"typ_name": "U-Turn Arrow", "filename_key": "Q"},
    "DEFAULT": {"typ_name": "Unknown Arrow", "filename_key": "DEFAULT"}
}

WORD_STYLES = {
    "DEFAULT": {"color": "black", "weight": "bold", "fontsize": 8, "zorder": 12},
    #possible to customise
    # "SLOW": {"color": "red", ...}
}


#Helper Functions

def get_feature_type(feature, dataset_name):
    props = feature.get('properties', {})

    if 'TYP_CD' in props: return props['TYP_CD']

    if dataset_name == "Kerb Markings": return "Generic Kerb"

    if dataset_name == "Word Markings":
        return props.get('DESC_TXT', 'Unknown Word')

    return "Unknown"


def get_feature_name(feature, dataset_name, typ_cd):
    props = feature.get('properties', {})
    if 'TYP_NAM' in props: return props['TYP_NAM']
    if dataset_name == "Arrow Markings" and typ_cd in ARROW_STYLES: return ARROW_STYLES[typ_cd]['typ_name']

    if dataset_name == "Word Markings": return typ_cd

    return f"Type {typ_cd}"


def get_filtered_cache_key(dataset_id, lat, lon, radius):
    key = f"{dataset_id}_{lat:.5f}_{lon:.5f}_{radius}m.json"
    return os.path.join(CACHE_DIR, key)


def load_filtered_features(cache_key, dataset_name):
    try:
        if os.path.exists(cache_key):
            with open(cache_key, 'r') as f:
                features = json.load(f)
            print(f"Cache Hit for {dataset_name}. Loaded {len(features)} features.")
            return features
        print(f"Cache Miss for {dataset_name}.")
        return None
    except Exception as e:
        print(f"Error loading cache for {dataset_name}: {e}. Will re-download.")
        return None


def save_filtered_features(cache_key, features):
    try:
        with open(cache_key, 'w') as f:
            json.dump(features, f)
        print(f"Saved {len(features)} features to cache: {cache_key}")
    except Exception as e:
        print(f"Error saving cache: {e}")


def get_geojson_data(dataset_id, dataset_name):
    print(f"1. Fetching GEOJSON download URL for {dataset_name}...")
    try:
        url = DOWNLOAD_URL_ENDPOINT.format(dataset_id=dataset_id)
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()

        if json_data.get('code') != 0:
            raise Exception(f"API Error for {dataset_name}: {json_data.get('errMsg', 'Unknown API error')}")

        download_url = json_data['data']['url']
        print(f"   Download URL retrieved. Downloading full GEOJSON data...")

        geojson_response = requests.get(download_url)
        geojson_response.raise_for_status()
        return geojson_response.json()

    except requests.exceptions.RequestException as e:
        print(f"   A network/API error occurred while fetching {dataset_name}: {e}")
        return None
    except Exception as e:
        print(f"   An error occurred during {dataset_name} data fetching: {e}")
        return None


def get_first_coord(geometry):
    geom_type = geometry.get('type')
    coords = geometry.get('coordinates', [])

    if geom_type == 'Point':
        return coords
    elif geom_type == 'LineString':
        return coords[0] if coords else None
    elif geom_type == 'MultiLineString':
        return coords[0][0] if coords and coords[0] else None
    elif geom_type == 'Polygon' and coords and coords[0]:
        return coords[0][0] if coords[0] else None
    elif geom_type == 'MultiPolygon' and coords and coords[0] and coords[0][0]:
        return coords[0][0][0] if coords[0][0] else None
    return None


def extract_all_coords_from_feature(geometry):
    geom_type = geometry.get('type')
    coords = geometry.get('coordinates', [])

    if geom_type == 'Point':
        yield coords
    elif geom_type == 'LineString':
        for coord in coords: yield coord
    elif geom_type == 'MultiLineString':
        for line_segment in coords:
            for coord in line_segment: yield coord
    elif geom_type == 'Polygon' and coords and coords[0]:
        for coord in coords[0]: yield coord
    elif geom_type == 'MultiPolygon' and coords and coords[0] and coords[0][0]:
        for coord in coords[0][0]: yield coord


def filter_features_by_radius(full_geojson, target_lat, target_lon, radius_meters, dataset_name):
    target_point = (target_lat, target_lon)
    filtered_features = []

    if dataset_name == "Kerb Markings":
        pre_check_radius = KERB_PRE_CHECK_RADIUS_METERS
    else:
        pre_check_radius = PRE_CHECK_RADIUS_METERS

    print(f"Filtering {dataset_name}. Pre-check: {pre_check_radius}m. Final: {radius_meters}m.")

    for feature in full_geojson.get('features', []):
        geometry = feature.get('geometry')
        if not geometry: continue

        first_coord = get_first_coord(geometry)
        if first_coord and len(first_coord) >= 2:
            lon_first, lat_first = first_coord[0], first_coord[1]
            distance_to_start = geodesic(target_point, (lat_first, lon_first)).meters
            if distance_to_start > pre_check_radius: continue

        is_near = False
        for coord in extract_all_coords_from_feature(geometry):
            if len(coord) < 2: continue
            lon, lat = coord[0], coord[1]
            distance = geodesic(target_point, (lat, lon)).meters
            if distance <= radius_meters:
                is_near = True
                break

        if is_near: filtered_features.append(feature)

    print(f"Filtered down to {len(filtered_features)} {dataset_name} features.")
    return filtered_features


def analyze_dataset_types(full_geojson, dataset_name, styles_dict):
    types = {}
    for feature in full_geojson.get('features', []):
        typ_cd = get_feature_type(feature, dataset_name)
        types[typ_cd] = types.get(typ_cd, 0) + 1

    print(f"\n--- {dataset_name} Type Analysis ({len(full_geojson.get('features', []))} Total) ---")
    sorted_types = sorted(types.items(), key=lambda item: item[1], reverse=True)

    # Print header
    print(f"{'Count':<8} | Feature Name / Type Code")
    print("-" * 40)

    for typ_cd, count in sorted_types:
        name = typ_cd if dataset_name == "Word Markings" else styles_dict.get(typ_cd, {}).get('typ_name', 'N/A')
        print(f"{count:<8} | {name}")
    print("----------------------------------------------------------------")


def preload_arrow_images():
    os.makedirs(ARROW_IMAGE_DIR, exist_ok=True)
    for key, filename in ARROW_IMAGES.items():
        filepath = os.path.join(ARROW_IMAGE_DIR, filename)
        if not os.path.exists(filepath):
            try:
                plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, key, ha='center', va='center', fontsize=50)
                plt.axis('off')
                plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
                plt.close()
            except Exception:
                continue
        IMAGE_CACHE[key] = filepath



def draw_road_features(lane_features, kerb_features, arrow_features, word_features, target_lat, target_lon,
                       radius_meters):
    print(f"\n3. Generating road plot with fixed limits and simplified alignment...")

    if not (lane_features or kerb_features or arrow_features or word_features):
        print("   WARNING: No features available to draw. Plot will show target area only.")

    lat_deg_per_meter = 1 / METERS_PER_DEGREE_LAT
    approx_radius_deg = radius_meters * lat_deg_per_meter
    PLOT_LIMIT_DEG = 1.5 * approx_radius_deg

    shifted_target_lon = target_lon + LON_SHIFT
    shifted_target_lat = target_lat + LAT_SHIFT

    legend_details = {}
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal', adjustable='box')

    def plot_line_features(features, styles_dict, is_kerb=False, dataset_name=""):
        prefix = "KERB " if is_kerb else "LANE "
        for feature in features:
            geometry = feature.get('geometry')
            coordinates = geometry.get('coordinates', [])
            geom_type = geometry.get('type', 'LineString')

            flat_coords = []
            if geom_type == 'MultiLineString':
                for line_segment in coordinates: flat_coords.extend(line_segment)
            elif geom_type == 'LineString' or geom_type == 'Point':
                flat_coords = coordinates
            elif geom_type == 'Polygon' and coordinates and coordinates[0]:
                flat_coords = coordinates[0]
            elif geom_type == 'MultiPolygon' and coordinates and coordinates[0] and coordinates[0][0]:
                flat_coords = coordinates[0][0]

            if not flat_coords: continue

            lons = [coord[0] + LON_SHIFT for coord in flat_coords]
            lats = [coord[1] + LAT_SHIFT for coord in flat_coords]

            typ_cd = get_feature_type(feature, dataset_name)
            style = styles_dict.get(typ_cd, styles_dict['DEFAULT'])
            typ_nam_raw = get_feature_name(feature, dataset_name, typ_cd)
            typ_nam = prefix + typ_nam_raw
            legend_key = f"{prefix}_{typ_cd}"

            if legend_key not in legend_details:
                ax.plot(lons, lats, label=typ_nam, **style)
                legend_details[legend_key] = (typ_nam, style, 'line')
            else:
                ax.plot(lons, lats, **style)

    def plot_arrow_features(features, styles_dict, dataset_name="Arrow Markings"):
        prefix = "ARROW "
        BASE_ZOOM = 0.05

        for feature in features:
            props = feature.get('properties', {})
            geometry = feature.get('geometry')
            if not geometry or geometry.get('type') != 'Point': continue
            coordinates = geometry.get('coordinates', [])
            if not coordinates or len(coordinates) < 2: continue

            lon_orig, lat_orig = coordinates[0], coordinates[1]
            typ_cd = get_feature_type(feature, dataset_name)
            style = styles_dict.get(typ_cd, styles_dict['DEFAULT'])

            bearing = props.get('BEARG_NUM', 0)
            final_rotation_deg = -bearing

            lon = lon_orig + LON_SHIFT
            lat = lat_orig + LAT_SHIFT

            filename_key = style.get('filename_key', 'DEFAULT')
            filepath = IMAGE_CACHE.get(filename_key)

            if filepath is None: continue

            try:
                pil_img = Image.open(filepath).convert("RGBA")

                rotated_img_pil = pil_img.rotate(final_rotation_deg, expand=True, resample=Image.Resampling.BICUBIC)

                arrow_img_rotated_array = np.asarray(rotated_img_pil)
                image_box = OffsetImage(arrow_img_rotated_array, zoom=BASE_ZOOM)
                image_box.image.set_clip_on(False)

                ab = AnnotationBbox(
                    image_box, (lon, lat), frameon=False, xycoords='data', boxcoords='data',
                    bboxprops={"zorder": 11}
                )

                ax.add_artist(ab)

            except Exception as e:
                print(f"   Error arrow {typ_cd}: {e}")
                continue

            typ_nam = prefix + style.get('typ_name', f"Type {typ_cd}")
            legend_key = f"{prefix}_{typ_cd}"
            if legend_key not in legend_details:
                legend_details[legend_key] = (typ_nam, {"color": "black"}, 'point', filename_key)

    def plot_word_features(features, styles_dict, dataset_name="Word Markings"):
        prefix = "WORD "
        for feature in features:
            props = feature.get('properties', {})
            geometry = feature.get('geometry')
            if not geometry or geometry.get('type') != 'Point': continue
            coordinates = geometry.get('coordinates', [])
            if not coordinates or len(coordinates) < 2: continue

            lon = coordinates[0] + LON_SHIFT
            lat = coordinates[1] + LAT_SHIFT

            word_text = props.get('DESC_TXT', 'UNK')
            bearing = props.get('BEARG_NUM', 0) + 90

            text_rotation = 90 - bearing

            style = styles_dict.get('DEFAULT').copy()

            ax.text(lon, lat, word_text,
                    rotation=text_rotation,
                    ha='center', va='center',
                    color=style['color'],
                    weight=style['weight'],
                    fontsize=4,
                    zorder=style['zorder'],
                    clip_on=True)

            legend_key = "WORD_MARKING"
            if legend_key not in legend_details:
                legend_details[legend_key] = ("Road Text Marking", style, 'text')

    plot_line_features(kerb_features, KERB_STYLES, is_kerb=True, dataset_name="Kerb Markings")
    plot_line_features(lane_features, LANE_STYLES, is_kerb=False, dataset_name="Lane Markings")
    plot_arrow_features(arrow_features, ARROW_STYLES, dataset_name="Arrow Markings")
    plot_word_features(word_features, WORD_STYLES, dataset_name="Word Markings")  # NEW

    ax.plot(shifted_target_lon, shifted_target_lat, 'ro', markersize=8, zorder=12, label=f'Target GPS')
    radius_circle = Circle((shifted_target_lon, shifted_target_lat), approx_radius_deg,
                           color='blue', linestyle='--', fill=False, alpha=0.5, zorder=9,
                           label=f'{radius_meters}m Radius')
    ax.add_patch(radius_circle)

    ax.set_xlim(shifted_target_lon - PLOT_LIMIT_DEG, shifted_target_lon + PLOT_LIMIT_DEG)
    ax.set_ylim(shifted_target_lat - PLOT_LIMIT_DEG, shifted_target_lat + PLOT_LIMIT_DEG)
    ax.set_xlabel(f"Longitude")
    ax.set_ylabel(f"Latitude")
    ax.set_title(f"LTA Road Features within {radius_meters}m (WGS-84)", fontsize=16)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_FILE, dpi=400)
    plt.close(fig)
    print(f"   Combined road plot successfully saved to **{OUTPUT_PNG_FILE}**")

    num_plotted_items = len(legend_details)
    fig_legend = plt.figure(figsize=(12, num_plotted_items * 0.5 + 2))
    ax_legend = fig_legend.add_subplot(111)
    handles = []
    labels = []
    sorted_keys = sorted(legend_details.keys())

    for key in sorted_keys:
        detail = legend_details[key]
        style_type = detail[2]

        if style_type == 'line':
            dummy_item = plt.Line2D([0], [0], color=detail[1]['color'], linewidth=detail[1]['linewidth'],
                                    linestyle=detail[1]['linestyle'])
            labels.append(f"{key.split('_')[1]} - {detail[0]}")
        elif style_type == 'point':
            stored_filename_key = detail[3]
            dummy_item = plt.Line2D([0], [0], linestyle='None', marker='s', markersize=8, markerfacecolor='white',
                                    markeredgecolor='black')
            filename = ARROW_IMAGES.get(stored_filename_key, "N/A")
            labels.append(f"{key.split('_')[1]} - {detail[0]} (Image: {filename})")
        elif style_type == 'text':
            dummy_item = plt.Line2D([0], [0], marker='$T$', markersize=10, color='black', linestyle='None')
            labels.append(f"{detail[0]}")

        handles.append(dummy_item)

    ax_legend.legend(handles, labels, loc='center', title="Road Feature Types", fontsize=10, title_fontsize=12)
    ax_legend.axis('off')
    fig_legend.tight_layout()
    fig_legend.savefig(LEGEND_OUTPUT_FILE, dpi=300)
    plt.close(fig_legend)
    print(f"   Legend saved separately to **{LEGEND_OUTPUT_FILE}**")



if __name__ == "__main__":
    print("--- LTA Road Feature Visualizer and Analyzer ---")
    preload_arrow_images()


    def get_processed_features(dataset_id, display_name, styles=None):
        cache_key = get_filtered_cache_key(dataset_id, TARGET_LAT, TARGET_LON, RADIUS_METERS)

        features = load_filtered_features(cache_key, display_name)

        if features is None:
            data = get_geojson_data(dataset_id, display_name)
            if data:
                if styles:
                    analyze_dataset_types(data, display_name, styles)

                features = filter_features_by_radius(
                    data, TARGET_LAT, TARGET_LON, RADIUS_METERS, display_name
                )
                save_filtered_features(cache_key, features)

        return features

    datasets = [
        ("lane", LANE_DATASET_ID, "Lane Markings", None),
        ("kerb", KERB_DATASET_ID, "Kerb Markings", None),
        ("arrow", ARROW_DATASET_ID, "Arrow Markings", None),
        ("word", WORD_DATASET_ID, "Word Markings", WORD_STYLES),
    ]

    results = {}

    for var_name, d_id, d_name, d_styles in datasets:
        results[var_name] = get_processed_features(d_id, d_name, d_styles)

    filtered_lane_features = results["lane"]
    filtered_kerb_features = results["kerb"]
    filtered_arrow_features = results["arrow"]
    filtered_word_features = results["word"]

    draw_road_features(
        filtered_lane_features or [],
        filtered_kerb_features or [],
        filtered_arrow_features or [],
        filtered_word_features or [],
        TARGET_LAT, TARGET_LON,
        RADIUS_METERS
    )

    print("\n--- Execution Complete ---")
