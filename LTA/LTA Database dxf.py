import requests
import json
import math
import os
from geopy.distance import geodesic
import re
import ezdxf
from ezdxf.math import offset_vertices_2d, Vec2

#the only two user inputs
TARGET_LAT, TARGET_LON = 1.316920, 103.892253
RADIUS_METERS = 50

#there's in built code to shift the entire thing laterally ot longitudinally, but this is not very important
LAT_SHIFT = 0.0
LON_SHIFT = 0.0


PRE_CHECK_RADIUS_METERS = RADIUS_METERS + 150
KERB_PRE_CHECK_RADIUS_METERS = RADIUS_METERS + 1000

# API Configuration
LANE_DATASET_ID = "d_fa71cc0c433275f4d2b0133358cc4fbf"
KERB_DATASET_ID = "d_45d62178079823aea49a2a7d3d7803c7"
ARROW_DATASET_ID = "d_2771fb949ceea676dd3eecade3830c5b"
WORD_DATASET_ID = "d_dfd0b6e8d5e82643d8278690e576a523"
LAMPPOST_DATASET_ID = "d_ca109de3e83efdd9a10bc5f3dda70a98"
TRAFFIC_LIGHT_DATASET_ID = "d_44cc2bba6131f9057fb92f367eba69ed"  # NEW Traffic Light API

#the format for queries
BASE_URL = "https://api-open.data.gov.sg/v1/public/api/datasets/"
DOWNLOAD_URL_ENDPOINT = f"{BASE_URL}{{dataset_id}}/poll-download"

CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)
OUTPUT_DXF_FILE = "road_features_faro_zone.dxf"

#manually tuning for different .dxf arrows
BLOCK_SCALES = {
    "ARROW_STRAIGHT": 0.025,
    "ARROW_RIGHT": 0.025,
    "ARROW_LEFT": 0.025,
    "ARROW_U_TURN": 0.025,
    "CON_LEFT": 0.025,
    "CON_RIGHT": 0.025,
    "ARROW_STRAIGHT_RIGHT": 0.002,
    "ARROW_STRAIGHT_LEFT": 0.002,
    "ARROW_LEFT_RIGHT": 0.025,
    "ARROW_STRAIGHT_LEFT_RIGHT": 0.025,
    "TRAFFIC_ARROW": 0.5,
    "DEFAULT": 0.025,
}
#radius controls below
#this controls poles for specifically for beacons
TL_ARROW_STEM_LENGTH = 1
TL_POLE_WIDTH = 0.1
#this controls arrows for the traffic lights
STEM_WIDTH = 0.1
STEM_START = 0.2
STEM_END = 3
ARROW_WIDTH = 0.6
ARROW_LENGTH = 1

#converts to meters for easier computation
METERS_PER_DEGREE_LAT = 111111.0
COS_TARGET_LAT = math.cos(math.radians(TARGET_LAT))
METERS_PER_DEGREE_LON = METERS_PER_DEGREE_LAT * COS_TARGET_LAT

#more manual dictionaries. color type can be changed. Search ".dxf colours". linetype and lineweight seems to be ignored by FARO Zone 3D
LANE_STYLES = {
    "A": {"layer": "LANE_DASHED_WHITE_1", "color": 7, "linetype": "DASHED", "lineweight": 25},
    "A1": {"layer": "LANE_DASHED_WHITE_1_THICK", "color": 7, "linetype": "DASHED", "lineweight": 50},
    "A2": {"layer": "LANE_DASHED_WHITE_2", "color": 7, "linetype": "DASHED", "lineweight": 25},
    "A3": {"layer": "LANE_DASHED_YELLOW_1", "color": 2, "linetype": "DASHED", "lineweight": 35},
    "A4": {"layer": "LANE_SOLID_WHITE_2_THIN", "color": 7, "linetype": "CONTINUOUS", "lineweight": 18},
    "A5": {"layer": "LANE_DASHED_WHITE_3", "color": 7, "linetype": "DASHED", "lineweight": 25},
    "A6": {"layer": "LANE_RED_DASHED_3", "color": 1, "linetype": "DASHED", "lineweight": 35},
    "A7": {"layer": "LANE_RED_DASHED_4", "color": 1, "linetype": "DASHED", "lineweight": 40},
    "A8": {"layer": "LANE_SOLID_WHITE_4_THICK", "color": 7, "linetype": "CONTINUOUS", "lineweight": 50},
    "B": {"layer": "LANE_DASHED_WHITE_4", "color": 7, "linetype": "DASHED2", "lineweight": 25},
    "B1": {"layer": "LANE_DASHED_WHITE_5", "color": 7, "linetype": "DASHED2", "lineweight": 25},
    "B2": {"layer": "LANE_DASHED_WHITE_6", "color": 7, "linetype": "DASHED", "lineweight": 30},
    "C": {"layer": "LANE_DASHED_WHITE_7", "color": 7, "linetype": "DASHED", "lineweight": 25},
    "C1": {"layer": "LANE_DASHED_WHITE_8", "color": 7, "linetype": "DASHED", "lineweight": 35},
    "D": {"layer": "LANE_DASHED_WHITE_9", "color": 7, "linetype": "DASHED", "lineweight": 25},
    "D1": {"layer": "LANE_DOTTED_WHITE_1", "color": 7, "linetype": "DOTTED", "lineweight": 25},
    "J": {"layer": "LANE_SOLID_WHITE_2", "color": 7, "linetype": "CONTINUOUS", "lineweight": 30},
    "R": {"layer": "LANE_DASHED_WHITE_10", "color": 7, "linetype": "DASHED", "lineweight": 30},
    "N": {"layer": "LANE_SOLID_BROWN_2", "color": 2, "linetype": "CONTINUOUS", "lineweight": 25},
    "E": {"layer": "LANE_DASHED_WHITE_11", "color": 7, "linetype": "DASHED", "lineweight": 40},
    "F": {"layer": "LANE_SOLID_WHITE_1_THIN", "color": 7, "linetype": "CONTINUOUS", "lineweight": 18},
    "G": {"layer": "LANE_SOLID_WHITE_1", "color": 7, "linetype": "CONTINUOUS", "lineweight": 25},
    "H": {"layer": "LANE_SOLID_WHITE_1_THIN_2", "color": 7, "linetype": "CONTINUOUS", "lineweight": 18},
    "I": {"layer": "LANE_SOLID_BROWN_3", "color": 2, "linetype": "CONTINUOUS", "lineweight": 25},
    "K": {"layer": "LANE_SOLID_WHITE_1_THIN_3", "color": 7, "linetype": "CONTINUOUS", "lineweight": 18},
    "L": {"layer": "LANE_SOLID_BROWN_4", "color": 2, "linetype": "CONTINUOUS", "lineweight": 40},
    "M": {"layer": "LANE_SOLID_WHITE_2_THICK", "color": 7, "linetype": "CONTINUOUS", "lineweight": 35},
    "O": {"layer": "LANE_DOTTED_WHITE_2", "color": 7, "linetype": "DOTTED", "lineweight": 25},
    "P": {"layer": "LANE_DOTTED_WHITE_3", "color": 7, "linetype": "DOTTED", "lineweight": 25},
    "Q": {"layer": "LANE_SOLID_RED_5", "color": 1, "linetype": "CONTINUOUS", "lineweight": 50},
    "Q1": {"layer": "LANE_SOLID_RED_3", "color": 1, "linetype": "CONTINUOUS", "lineweight": 35},
    "Q2": {"layer": "LANE_DOTTED_RED_3", "color": 1, "linetype": "DOTTED", "lineweight": 30},
    "S": {"layer": "LANE_DOTTED_WHITE_4", "color": 7, "linetype": "DOTTED", "lineweight": 30},
    "T": {"layer": "LANE_DASHED_WHITE_12", "color": 7, "linetype": "DASHED", "lineweight": 30},
    "U": {"layer": "LANE_DASHED_WHITE_13", "color": 7, "linetype": "DASHED", "lineweight": 18},
    "V": {"layer": "LANE_DASH_DOT_WHITE", "color": 7, "linetype": "DASHDOT", "lineweight": 40},
    "W": {"layer": "LANE_DOUBLE_WHITE", "color": 7, "draw_type": "DOUBLE", "separation": 0.15, "linetype": "CONTINUOUS",
          "lineweight": 25},
    "X": {"layer": "LANE_DOTTED_WHITE_5", "color": 7, "linetype": "DOTTED", "lineweight": 30},
    "Y": {"layer": "LANE_SOLID_WHITE_1_THIN_4", "color": 7, "linetype": "CONTINUOUS", "lineweight": 18},
    "Z": {"layer": "LANE_DOTTED_WHITE_6", "color": 7, "linetype": "DOTTED", "lineweight": 25},
    "DEFAULT": {"layer": "LANE_UNKNOWN", "color": 8, "linetype": "CONTINUOUS", "lineweight": 13}
}

KERB_STYLES = {
    "Generic Kerb": {"layer": "KERB_GENERIC", "color": 1, "linetype": "CONTINUOUS", "lineweight": 30},
    "DEFAULT": {"layer": "KERB_UNKNOWN", "color": 8, "linetype": "CONTINUOUS", "lineweight": 13}
}
#typ_name is what is in the database.
ARROW_STYLES = {
    "A": {"typ_name": "R-Turn", "layer": "ARROW_R_TURN", "color": 7, "block_name": "ARROW_RIGHT"},
    "D": {"typ_name": "Straight", "layer": "ARROW_STRAIGHT", "color": 7, "block_name": "ARROW_STRAIGHT"},
    "E": {"typ_name": "L-Turn", "layer": "ARROW_L_TURN", "color": 7, "block_name": "ARROW_LEFT"},
    "G": {"typ_name": "L-Conv", "layer": "ARROW_L_CONV", "color": 7, "block_name": "CON_LEFT"},
    "H": {"typ_name": "R-Conv", "layer": "ARROW_R_CONV", "color": 7, "block_name": "CON_RIGHT"},
    "L": {"typ_name": "PT-R-Turn", "layer": "ARROW_PT_R_TURN", "color": 7, "block_name": "ARROW_RIGHT"},
    "M": {"typ_name": "PT-L-Turn", "layer": "ARROW_PT_L_TURN", "color": 7, "block_name": "ARROW_LEFT"},
    "N": {"typ_name": "PT-Straight", "layer": "ARROW_PT_STRAIGHT", "color": 7, "block_name": "ARROW_STRAIGHT"},
    "B": {"typ_name": "St/Lt", "layer": "ARROW_ST_LT", "color": 7, "block_name": "ARROW_STRAIGHT_LEFT"},
    "C": {"typ_name": "St/Rt", "layer": "ARROW_ST_RT", "color": 7, "block_name": "ARROW_STRAIGHT_RIGHT"},
    "F": {"typ_name": "Lt/Rt", "layer": "ARROW_LT_RT", "color": 7, "block_name": "ARROW_LEFT_RIGHT"},
    "I": {"typ_name": "St/Lt/Rt", "layer": "ARROW_ST_LT_RT", "color": 7, "block_name": "ARROW_STRAIGHT_LEFT_RIGHT"},
    "J": {"typ_name": "St/Rt-Yel", "layer": "ARROW_ST_RT_YELLOW", "color": 2, "block_name": "ARROW_STRAIGHT_RIGHT"},
    "K": {"typ_name": "PT-St/Rt", "layer": "ARROW_PT_ST_RT", "color": 7, "block_name": "ARROW_STRAIGHT_RIGHT"},
    "O": {"typ_name": "PT-Lt/Rt", "layer": "ARROW_PT_LT_RT", "color": 7, "block_name": "ARROW_LEFT_RIGHT"},
    "P": {"typ_name": "PT-St/Lt", "layer": "ARROW_PT_ST_LT", "color": 7, "block_name": "ARROW_STRAIGHT_LEFT"},
    "Q": {"typ_name": "U-Turn", "layer": "ARROW_U_TURN", "color": 7, "block_name": "ARROW_U_TURN"},
    "DEFAULT": {"typ_name": "UNK_ARROW", "layer": "ARROW_UNKNOWN", "color": 8, "block_name": "ARROW_STRAIGHT"}
}

WORD_STYLES = {
    "DEFAULT": {"layer": "WORD_MARKING", "color": 7},
}


LAMPPOST_STYLES = {
    "DEFAULT": {"layer": "LAMPPOSTS", "color": 253, "radius": 0.5, "linetype": "CONTINUOUS", "lineweight": 20},
}

#radius controls
TRAFFIC_LIGHT_STYLES = {
    "DEFAULT": {"layer": "TRAFFIC_LIGHTS", "color": 4, "radius": 0.5, "linetype": "CONTINUOUS", "lineweight": 20},
}


#different helper functions for different data types
def get_feature_type(feature, dataset_name):
    props = feature.get('properties', {})
    if 'TYP_CD' in props: return props['TYP_CD']
    if dataset_name == "Kerb Markings": return "Generic Kerb"
    if dataset_name == "Word Markings": return props.get('DESC_TXT', 'Unknown Word')
    if dataset_name == "Traffic Lights": return props.get('TYP_NAM', 'Unknown TL')  # Use TYPE NAME for traffic lights
    return "Unknown"


def get_feature_name(feature, dataset_name, typ_cd):
    props = feature.get('properties', {})
    if 'TYP_NAM' in props: return props['TYP_NAM']
    if dataset_name == "Arrow Markings" and typ_cd in ARROW_STYLES: return ARROW_STYLES[typ_cd]['typ_name']
    if dataset_name == "Word Markings": return typ_cd
    if dataset_name == "Traffic Lights": return typ_cd
    return f"Type {typ_cd}"

#caching for faster reruns -- useful when changing visualisation only
def get_filtered_cache_key(dataset_id, lat, lon, radius):
    key = f"{dataset_id}_{lat:.5f}_{lon:.5f}_{radius}m.json"
    return os.path.join(CACHE_DIR, key)


def load_filtered_features(cache_key, dataset_name):
    try:
        if os.path.exists(cache_key):
            with open(cache_key, 'r') as f: return json.load(f)
        return None
    except Exception:
        return None


def save_filtered_features(cache_key, features):
    try:
        with open(cache_key, 'w') as f:
            json.dump(features, f)
    except Exception:
        pass

#the API request code
def get_geojson_data(dataset_id, dataset_name):
    try:
        url = DOWNLOAD_URL_ENDPOINT.format(dataset_id=dataset_id)
        response = requests.get(url)
        json_data = response.json()
        if 'code' in json_data and json_data['code'] != 0:
            print(f"API Error fetching {dataset_name}: {json_data['errMsg']}")
            return None
        download_url = json_data['data']['url']
        geojson_response = requests.get(download_url)
        return geojson_response.json()
    except Exception as e:
        print(f"Error fetching {dataset_name}: {e}")
        return None

#first coordinate logic
def get_first_coord(geometry):
    geom_type = geometry.get('type')
    coords = geometry.get('coordinates', [])
    if geom_type == 'Point':
        return coords
    elif geom_type == 'LineString':
        return coords[0] if coords else None
    return None

#all coordinates
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

#checks the first coord of an object if within radius + 150, checks all coords. If any coord is within radius, saves it
def filter_features_by_radius(full_geojson, target_lat, target_lon, radius_meters, dataset_name):
    target_point = (target_lat, target_lon)
    filtered_features = []
    pre_check_radius = KERB_PRE_CHECK_RADIUS_METERS if dataset_name == "Kerb Markings" else PRE_CHECK_RADIUS_METERS

    for feature in full_geojson.get('features', []):
        geometry = feature.get('geometry')
        if not geometry: continue
        first_coord = get_first_coord(geometry)
        if first_coord and len(first_coord) >= 2:
            if geodesic(target_point, (first_coord[1], first_coord[0])).meters > pre_check_radius: continue
        is_near = False
        for coord in extract_all_coords_from_feature(geometry):
            if len(coord) < 2: continue
            dist_lat = coord[1] + LAT_SHIFT
            dist_lon = coord[0] + LON_SHIFT
            if geodesic(target_point, (dist_lat, dist_lon)).meters <= radius_meters:
                is_near = True
                break
        if is_near: filtered_features.append(feature)
    return filtered_features


#dxf conversion code

def convert_wgs_to_local_xy(lon, lat):
    x_m = (lon + LON_SHIFT - TARGET_LON) * METERS_PER_DEGREE_LON
    y_m = (lat + LAT_SHIFT - TARGET_LAT) * METERS_PER_DEGREE_LAT
    return (x_m, y_m)

#maybe you can edit this to get it to import as dashed?
def setup_dxf_linetypes(doc):
    if "DASHED" not in doc.linetypes:
        doc.linetypes.new("DASHED", dxfattribs={"description": "Dashed", "pattern": [1.0, 0.5, -0.25]})
    if "DASHED2" not in doc.linetypes:
        doc.linetypes.new("DASHED2", dxfattribs={"description": "Dashed (2x)", "pattern": [2.0, 1.0, -0.5]})
    if "DOTTED" not in doc.linetypes:
        doc.linetypes.new("DOTTED", dxfattribs={"description": "Dotted", "pattern": [0.5, 0.0, -0.25]})
    if "DASHDOT" not in doc.linetypes:
        doc.linetypes.new("DASHDOT", dxfattribs={"description": "Dash Dot", "pattern": [1.0, 0.5, -0.25, 0.0, -0.25]})


def create_dxf_layers(doc, styles_list):
    for style in styles_list:
        layer_name = style['layer']
        if layer_name not in doc.layers:
            doc.layers.new(layer_name, dxfattribs={
                'color': style['color'],
                'linetype': style.get('linetype', 'CONTINUOUS'),
                'lineweight': style.get('lineweight', 13)
            })


def setup_arrow_blocks(doc):
    #this will explode if arrow blocks have more than one vector per file
    def load_external_geometry(block, dxf_filepath):
        print(f"Attempting to read custom block geometry from: {dxf_filepath}")
        try:
            source_doc = ezdxf.readfile(dxf_filepath)
        except IOError:
            print(f"Warning: Could not read {dxf_filepath}. Skipping external load.")
            return False


        for entity in source_doc.modelspace():
            block.add_entity(entity.copy())
        print(f"Successfully loaded block geometry from {dxf_filepath} using direct insertion.")
        return True

    ARROW_MAPPING = {
        'ARROW_STRAIGHT': 'Straight_Single.dxf',
        'ARROW_RIGHT': 'Right_Turn_Single.dxf',
        'ARROW_LEFT': 'Left_Turn_Single.dxf',
        'CON_LEFT': 'Left_Conversional.dxf',
        'CON_RIGHT': 'Right_Conversional.dxf',
        'ARROW_STRAIGHT_LEFT_RIGHT': 'Straight_Left_Right_Turn.dxf',
        'ARROW_STRAIGHT_RIGHT': 'Straight_Right_Turn.dxf',
        'ARROW_STRAIGHT_LEFT': 'Straight_Left_Turn.dxf',
        'ARROW_LEFT_RIGHT': 'Left_Right_Turn.dxf',
    }

    IMAGE_DIR = "custom_arrow_images"

    for block_name, filename in ARROW_MAPPING.items():
        blk = doc.blocks.new(name=block_name)
        dxf_filepath = os.path.join(IMAGE_DIR, filename)
        load_external_geometry(blk, dxf_filepath)

    #i have no idea what u turn arrows are
    blk = doc.blocks.new(name='ARROW_U_TURN')
    U_TURN_RIGHT_SHAFT_COORDS = [(0.35, -2.0), (0.65, -2.0), (0.65, 0.0), (0.35, 0.0)]
    U_TURN_TOP_BAR_COORDS = [(-0.65, 0.0), (0.65, 0.0), (0.65, 0.5), (-0.65, 0.5)]
    U_TURN_LEFT_HEAD_COORDS = [(-0.85, 0.0), (-0.15, 0.0), (-0.5, -1.0)]

    # Right shaft
    blk.add_lwpolyline(U_TURN_RIGHT_SHAFT_COORDS, close=True)
    blk.add_hatch(color=7).paths.add_polyline_path(U_TURN_RIGHT_SHAFT_COORDS, is_closed=True)
    # Top bar
    blk.add_lwpolyline(U_TURN_TOP_BAR_COORDS, close=True)
    blk.add_hatch(color=7).paths.add_polyline_path(U_TURN_TOP_BAR_COORDS, is_closed=True)
    # Left Head (pointing down)
    blk.add_lwpolyline(U_TURN_LEFT_HEAD_COORDS, close=True)
    blk.add_hatch(color=7).paths.add_polyline_path(U_TURN_LEFT_HEAD_COORDS, is_closed=True)


    #this is the arrow for traffic lights
    blk = doc.blocks.new(name='TRAFFIC_ARROW')

    TL_RADIUS = TRAFFIC_LIGHT_STYLES['DEFAULT']['radius']

    STEM_COORDS = [
        (-STEM_WIDTH / 2, STEM_START),
        (STEM_WIDTH / 2, STEM_START),
        (STEM_WIDTH / 2, STEM_END),
        (-STEM_WIDTH / 2, STEM_END)
    ]
    blk.add_lwpolyline(STEM_COORDS, close=True)
    blk.add_hatch(color=7).paths.add_polyline_path(STEM_COORDS, is_closed=True)

    TRAFFIC_HEAD_COORDS = [
        (-ARROW_WIDTH / 2, STEM_END),
        (ARROW_WIDTH / 2, STEM_END),
        (0.0, STEM_END + ARROW_LENGTH)
    ]
    blk.add_lwpolyline(TRAFFIC_HEAD_COORDS, close=True)
    blk.add_hatch(color=7).paths.add_polyline_path(TRAFFIC_HEAD_COORDS, is_closed=True)


#this does most of the work exporting to .dxf.
def export_road_features_to_dxf(lane_features, kerb_features, arrow_features, word_features, lampost_features,
                                traffic_light_features):
    print(f"\n3. Generating DXF file **{OUTPUT_DXF_FILE}**...")

    doc = ezdxf.new('R2010')
    setup_dxf_linetypes(doc)
    setup_arrow_blocks(doc)

    msp = doc.modelspace()
    all_styles = (
            list(LANE_STYLES.values()) +
            list(KERB_STYLES.values()) +
            list(ARROW_STYLES.values()) +
            list(WORD_STYLES.values()) +
            list(LAMPPOST_STYLES.values()) +
            list(TRAFFIC_LIGHT_STYLES.values())
    )
    create_dxf_layers(doc, all_styles)

    def draw_line_features(features, styles_dict, dataset_name):
        for feature in features:
            geometry = feature.get('geometry')
            coordinates = geometry.get('coordinates', [])
            geom_type = geometry.get('type', 'LineString')
            flat_coords = []
            if geom_type == 'MultiLineString':
                for line_segment in coordinates: flat_coords.extend(line_segment)
            elif geom_type == 'LineString':
                flat_coords = coordinates
            elif 'Polygon' in geom_type and coordinates:
                flat_coords = coordinates[0][0] if geom_type == 'MultiPolygon' else coordinates[0]

            if not flat_coords: continue

            typ_cd = get_feature_type(feature, dataset_name)
            style = styles_dict.get(typ_cd, styles_dict['DEFAULT'])
            layer = style['layer']
            draw_type = style.get('draw_type', 'SINGLE')

            points_xy = [Vec2(convert_wgs_to_local_xy(coord[0], coord[1])) for coord in flat_coords if len(coord) >= 2]
            if not points_xy: continue

            if draw_type == 'SINGLE':
                msp.add_lwpolyline(points_xy, dxfattribs={'layer': layer, 'color': style['color']})
            elif draw_type == 'DOUBLE':
                half_sep = style.get('separation', 0.2) / 2.0
                msp.add_lwpolyline(offset_vertices_2d(points_xy, offset=half_sep),
                                   dxfattribs={'layer': layer, 'color': style['color']})
                msp.add_lwpolyline(offset_vertices_2d(points_xy, offset=-half_sep),
                                   dxfattribs={'layer': layer, 'color': style['color']})

    draw_line_features(kerb_features, KERB_STYLES, "Kerb Markings")
    draw_line_features(lane_features, LANE_STYLES, "Lane Markings")

    def draw_point_features(features, styles_dict, dataset_name):
        for feature in features:
            props = feature.get('properties', {})
            geometry = feature.get('geometry')
            if not geometry or geometry.get('type') != 'Point': continue
            coordinates = geometry.get('coordinates', [])
            if not coordinates: continue

            typ_cd = get_feature_type(feature, dataset_name)
            style = styles_dict.get(typ_cd, styles_dict['DEFAULT'])
            layer = style['layer']

            x, y = convert_wgs_to_local_xy(coordinates[0], coordinates[1])

            if dataset_name == "Arrow Markings":
                # Arrow Blocks
                bearing = props.get('BEARG_NUM', 0)
                rotation_deg = -bearing

                block_name = style.get('block_name', 'ARROW_STRAIGHT')

                scale = BLOCK_SCALES.get(block_name, BLOCK_SCALES["DEFAULT"])

                msp.add_blockref(block_name, insert=(x, y), dxfattribs={
                    'layer': layer,
                    'rotation': rotation_deg,
                    'color': style['color'],
                    'xscale': scale,
                    'yscale': scale,
                    'zscale': scale
                })
            elif dataset_name == "Word Markings":
                # Text/Word Markings
                bearing = props.get('BEARG_NUM', 0)
                rotation_deg = 90 - bearing

                text_content = props.get('DESC_TXT', 'UNK')
                msp.add_text(text_content, dxfattribs={
                    'layer': layer, 'color': style['color'], 'height': 1.0, 'rotation': rotation_deg,
                    'insert': (x, y), 'halign': 1, 'valign': 2
                })

    draw_point_features(arrow_features, ARROW_STYLES, "Arrow Markings")
    draw_point_features(word_features, WORD_STYLES, "Word Markings")

    # --- Draw Lamp Posts (Solid Circles) ---
    def draw_lampost_features(features, styles_dict, dataset_name):
        style = styles_dict['DEFAULT']
        layer = style['layer']
        radius = style['radius']
        color = style['color']

        for feature in features:
            geometry = feature.get('geometry')
            coordinates = geometry.get('coordinates', [])

            if geometry.get('type') == 'Point' and coordinates:
                x, y = convert_wgs_to_local_xy(coordinates[0], coordinates[1])

                # Draw Circle
                msp.add_circle(center=(x, y), radius=radius, dxfattribs={'layer': layer, 'color': color})

                #fill
                hatch = msp.add_hatch(color=color, dxfattribs={'layer': layer})
                num_segments = 32
                vertices = []
                for i in range(num_segments):
                    angle = 2 * math.pi * i / num_segments
                    vx = x + radius * math.cos(angle)
                    vy = y + radius * math.sin(angle)
                    vertices.append((vx, vy))

                hatch.paths.add_polyline_path(vertices, is_closed=True)

                props = feature.get('properties', {})
                label = props.get('LAMPPOST_NUM', 'LP')

                msp.add_text(label, dxfattribs={
                    'layer': layer,
                    'color': 7,
                    'height': 1,
                    'insert': (x + radius * 1.5, y),
                    'halign': 0,
                    'valign': 2
                })

    draw_lampost_features(lamposts_filtered, LAMPPOST_STYLES, "Lamp Post")


    def draw_traffic_light_features(features, styles_dict, dataset_name):
        style = styles_dict['DEFAULT']
        layer = style['layer']
        radius = style['radius']
        color = style['color']

        pole_width = TL_POLE_WIDTH
        pole_length = TL_ARROW_STEM_LENGTH

        block_name = "TRAFFIC_ARROW"
        scale = BLOCK_SCALES.get(block_name, 0.5)

        for feature in features:
            props = feature.get('properties', {})
            geometry = feature.get('geometry')
            coordinates = geometry.get('coordinates', [])

            if geometry.get('type') == 'Point' and coordinates:
                x, y = convert_wgs_to_local_xy(coordinates[0], coordinates[1])
                bearing = props.get('BEARG_NUM', 0)
                rotation_deg = -bearing
                typ_name = props.get('TYP_NAM', 'Unknown TL')

                #beacons dont have direction so they have a pole instead of arrow
                is_not_directional = "Beacon" in typ_name or "PT" in typ_name


                msp.add_circle(center=(x, y), radius=radius, dxfattribs={'layer': layer, 'color': color})
                hatch = msp.add_hatch(color=color, dxfattribs={'layer': layer})

                num_segments = 32
                vertices = []
                for i in range(num_segments):
                    angle = 2 * math.pi * i / num_segments
                    vx = x + radius * math.cos(angle)
                    vy = y + radius * math.sin(angle)
                    vertices.append((vx, vy))

                hatch.paths.add_polyline_path(vertices, is_closed=True)

                if is_not_directional:
                    # Draw Pole
                    pole_coords = [
                        (x - pole_width / 2, y - radius),
                        (x + pole_width / 2, y - radius),
                        (x + pole_width / 2, y - radius - pole_length),
                        (x - pole_width / 2, y - radius - pole_length)
                    ]
                    msp.add_lwpolyline(pole_coords, close=True, dxfattribs={'layer': layer, 'color': 7})
                    pole_hatch = msp.add_hatch(color=7, dxfattribs={'layer': layer})
                    pole_hatch.paths.add_polyline_path(pole_coords, is_closed=True)
                else:
                    #draw arrow
                    msp.add_blockref(block_name, insert=(x, y), dxfattribs={
                        'layer': layer,
                        'rotation': rotation_deg,
                        'color': 7,
                        'xscale': scale,
                        'yscale': scale,
                        'zscale': scale
                    })

                #add text
                label = typ_name
                msp.add_text(label, dxfattribs={
                    'layer': layer,
                    'color': color,
                    'height': 1,
                    'insert': (x + radius + 0.1, y + radius + 0.1),
                    'halign': 0,
                    'valign': 2
                })

    draw_traffic_light_features(traffic_light_features, TRAFFIC_LIGHT_STYLES, "Traffic Lights")

    try:
        doc.saveas(OUTPUT_DXF_FILE)
        print(f"   DXF export complete: {OUTPUT_DXF_FILE}")
    except Exception as e:
        print(f"   ERROR saving DXF: {e}")


if __name__ == "__main__":
    print("--- LTA Road Feature DXF Exporter ---")

    # Define keys
    lane_key = get_filtered_cache_key(LANE_DATASET_ID, TARGET_LAT, TARGET_LON, RADIUS_METERS)
    kerb_key = get_filtered_cache_key(KERB_DATASET_ID, TARGET_LAT, TARGET_LON, RADIUS_METERS)
    arrow_key = get_filtered_cache_key(ARROW_DATASET_ID, TARGET_LAT, TARGET_LON, RADIUS_METERS)
    word_key = get_filtered_cache_key(WORD_DATASET_ID, TARGET_LAT, TARGET_LON, RADIUS_METERS)
    lampost_key = get_filtered_cache_key(LAMPPOST_DATASET_ID, TARGET_LAT, TARGET_LON, RADIUS_METERS)
    traffic_light_key = get_filtered_cache_key(TRAFFIC_LIGHT_DATASET_ID, TARGET_LAT, TARGET_LON, RADIUS_METERS)


    # Helper
    def process_dataset(dataset_id, dataset_name, cache_key, radius_meters):
        #check for cache
        features_filtered = load_filtered_features(cache_key, dataset_name)
        if features_filtered is not None:
            print(f"1. {dataset_name} data loaded from cache.")
            return features_filtered

        #request API if no cache
        print(f"1. Fetching raw {dataset_name} data...")
        features_raw = get_geojson_data(dataset_id, dataset_name)

        if features_raw:
            print(f"2. Filtering {dataset_name} data by radius ({radius_meters}m)...")
            features_filtered = filter_features_by_radius(features_raw, TARGET_LAT, TARGET_LON, radius_meters,
                                                          dataset_name)

            #save cache
            save_filtered_features(cache_key, features_filtered)
            return features_filtered

        print(f"   Warning: Could not fetch or load {dataset_name} data.")
        return []


    # Process all datasets
    lanes_filtered = process_dataset(LANE_DATASET_ID, "Lane Markings", lane_key, RADIUS_METERS)
    kerbs_filtered = process_dataset(KERB_DATASET_ID, "Kerb Markings", kerb_key, RADIUS_METERS)
    arrows_filtered = process_dataset(ARROW_DATASET_ID, "Arrow Markings", arrow_key, RADIUS_METERS)
    words_filtered = process_dataset(WORD_DATASET_ID, "Word Markings", word_key, RADIUS_METERS)
    lamposts_filtered = process_dataset(LAMPPOST_DATASET_ID, "Lamp Posts", lampost_key, RADIUS_METERS)
    traffic_lights_filtered = process_dataset(TRAFFIC_LIGHT_DATASET_ID, "Traffic Lights", traffic_light_key,
                                              RADIUS_METERS)

    # Export
    export_road_features_to_dxf(
        lanes_filtered,
        kerbs_filtered,
        arrows_filtered,
        words_filtered,
        lamposts_filtered,
        traffic_lights_filtered
    )
    print("\n--- Execution Complete ---")
