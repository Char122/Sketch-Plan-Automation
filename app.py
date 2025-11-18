
#run code with: streamlit run app.py


import streamlit as st #im using ver1.13.1
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import cv2 #pip install opencv-python
import os
import uuid




def invert_road_and_markings(image):
    #thresholds hardcoded for apple maps
    if image is None:
        print(f"Error: Could not load image")
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white_markings = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_markings = np.array([255, 255, 255], dtype=np.uint8)
    white_markings_mask = cv2.inRange(hsv, lower_white_markings, upper_white_markings)

    lower_road_grey = np.array([0, 0, 50], dtype=np.uint8)
    upper_road_grey = np.array([179, 50, 160], dtype=np.uint8)
    road_mask = cv2.inRange(hsv, lower_road_grey, upper_road_grey)

    road_mask_cleaned = cv2.bitwise_and(road_mask, road_mask, mask=cv2.bitwise_not(white_markings_mask))
    return road_mask_cleaned


if "state" not in st.session_state:
    st.session_state.state = "" #need to control streamlit refreshes to stop it from reloading everything

st.set_page_config(page_title="Traffic Accident Sketch Automation", layout="wide")
st.title("Traffic Accident Sketch")

if st.session_state.state == "": #this will not rerun when state is changed
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Use Apple Maps"):
            st.session_state.state = "Apple Maps"
    with col2:
        if st.button("Use 3D LiDAR (code not integrated)"):
            st.session_state.state = "3D LiDAR"
        #this doesn't actually do anything yet, the code that runs this pipeline is in a separate version

#section 1: takes file path of image and lets you crop it
if st.session_state.state == "Apple Maps":
    st.title("Apple Maps → Traffic Accident Sketch")

    def crop_image(image_np_array, roi_box):
        if image_np_array is None or roi_box is None:
            return None
        left, top, right, bottom = roi_box
        if left >= right or top >= bottom:
            st.warning("Invalid crop dimensions. Ensure right > left and bottom > top.")
            return image_np_array
        cropped_image = image_np_array[top:bottom, left:right]
        return cropped_image

    st.sidebar.header("1) Upload point cloud")

    tmp_path = st.sidebar.text_input(
        "Enter the full path to your .e57 file:",
        value=""
    )

    if not tmp_path or not os.path.isfile(tmp_path):
        st.sidebar.info("Please enter a valid path to your .e57 file on disk.")
        st.stop()

    # Save path to tmp_path (for compatibility with existing code)
    st.sidebar.success(f"Using local file: {tmp_path} ({os.path.getsize(tmp_path) / 1e9:.2f} GB)")

    st.sidebar.success(f"Saved uploaded file to {tmp_path}")

    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'masked_image' not in st.session_state:
        st.session_state.masked_image = None
    if 'cropped_image' not in st.session_state:
        st.session_state.cropped_image = None
    if 'roi_box' not in st.session_state:
        st.session_state.roi_box = None
    if 'current_image_for_canvas' not in st.session_state:
        st.session_state.current_image_for_canvas = None

    base_image_np = cv2.imread(tmp_path)
    st.session_state.precrop_image = base_image_np
    base_image = Image.fromarray(cv2.cvtColor(base_image_np, cv2.COLOR_BGR2RGB))
    new_height = math.floor(700 * base_image.size[1] / base_image.size[0]) #resizing to standardised width but height maintains aspect ratio
    base_image = base_image.resize((700, new_height))
    array = np.array(base_image)
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=base_image,
        update_streamlit=True,
        height=base_image.size[1],
        width=base_image.size[0],
        drawing_mode="rect",
        key="roi_canvas"
    )

    roi_box_from_canvas = None
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        for o in objects:
            if o.get("type") == "rect":
                left = o.get("left", 0)
                top = o.get("top", 0)
                width = o.get("width", 0)
                height = o.get("height", 0)
                roi_box_from_canvas = (
                    left, top, left + width, top + height
                )
                break

    if roi_box_from_canvas:
        st.session_state["roi_box"] = roi_box_from_canvas
        st.sidebar.success(f"Selected ROI: {st.session_state['roi_box']}")
        #preview of the selected ROI
        st.subheader("Selected ROI Preview")
        temp_cropped_preview = crop_image(st.session_state.precrop_image, st.session_state.roi_box)
        st.session_state.original_image = temp_cropped_preview
        if temp_cropped_preview is not None:
            st.image(temp_cropped_preview, caption="Temporary Cropped Preview", use_column_width=True)
    else:
        st.sidebar.info("No ROI selected yet. Draw a rectangle on the preview to select one.")

    if st.button("Confirm Crop and Proceed"):
        if st.session_state.roi_box is None:
            st.warning("Please select an ROI before proceeding.")
        else:
            st.session_state.cropped_image = crop_image(st.session_state.precrop_image, st.session_state.roi_box)
            if st.session_state.cropped_image is not None:
                st.success("Image cropped successfully!")

            else:
                st.error("Failed to crop image.")

            st.session_state.state = "calibrate_scale"  # Change state to next step so this code section is never rerun
            st.experimental_rerun()

#section 2: input calibration scale
if st.session_state.state == "calibrate_scale":
    st.header("📏 Step 5: Calibrate Distance Scale")

    if st.session_state.cropped_image is not None:
        st.image(st.session_state.cropped_image, caption="Draw a calibration arrow", use_column_width=True)

        st.write("Draw a line across a known real-world distance (e.g., width of lane, 3.5 m).")

        base_image_np = st.session_state.cropped_image
        rgb_array = cv2.cvtColor(base_image_np, cv2.COLOR_BGR2RGB)
        base_image_pil = Image.fromarray(rgb_array)
        width, height = base_image_pil.size

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="#00ff00",
            background_image=base_image_pil,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="line",
            key="calibration_canvas"
        )

        distance_m = st.number_input("Enter real-world distance (in meters):", min_value=0.0, step=0.1)

        if st.button("Confirm Calibration"):
            if canvas_result.json_data and "objects" in canvas_result.json_data and len(
                    canvas_result.json_data["objects"]) > 0:
                obj = canvas_result.json_data["objects"][-1]
                if obj["type"] == "line":
                    cx, cy = obj["left"], obj["top"]
                    x1, y1 = cx + obj["x1"], cy + obj["y1"]
                    x2, y2 = cx + obj["x2"], cy + obj["y2"]
                    pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance_m > 0 and pixel_distance > 0:
                        st.session_state.scale_px_per_m = pixel_distance / distance_m * 1350 / width #need to rescale for target width later on
                        st.success(f"✅ Calibration complete: 1 px = {st.session_state.scale_px_per_m:.5f} meters")
                        st.session_state.state = "cropped_view"
                        st.rerun()
                    else:
                        st.error("Please enter a valid distance and draw a valid line.")
            else:
                st.warning("Please draw a calibration line before confirming.")
    else:
        st.warning("No cropped image available. Please go back and crop again.")

#section 3: shows cropped view
if st.session_state.state == "cropped_view":
    st.header("4) Cropped Image Result")
    if st.session_state.cropped_image is not None:
        st.image(st.session_state.cropped_image, caption="Final Cropped Image", use_column_width=True)

        # You can now re-mask the cropped image or do other operations
        st.subheader("Masked Cropped Image")
        masked_cropped = invert_road_and_markings(st.session_state.cropped_image)
        # Check if masking was successful before setting state
        if masked_cropped is not None:
            st.session_state.mask = masked_cropped
            st.image(masked_cropped, caption="Masked Cropped Image", use_column_width=True)
            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("Proceed"):
                    st.session_state.state = "Edit"
                    st.experimental_rerun()
            with col2:
                if st.button("Go Back"):
                    st.session_state.state = "Apple Maps"
                    st.experimental_rerun()
        else:
            st.warning("Could not generate mask for the cropped image.")


    else:
        st.warning("No cropped image available. Please go back and crop an image.")
        if st.button("Go Back"):
            st.session_state.state = "Apple Maps"
            st.experimental_rerun()
#section 4: editor:
if "road_sketch_draft" not in st.session_state and "mask" in st.session_state and st.session_state.state == "Edit":
    def transform_mask_to_rgba(mask_input) -> Image.Image:

        if isinstance(mask_input, np.ndarray):
            mask_l = Image.fromarray(mask_input.astype(np.uint8))
        elif isinstance(mask_input, Image.Image):
            mask_l = mask_input
        else:
            raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(mask_input)}")

        mask_np = np.array(mask_l.convert('L'))
        h, w = mask_np.shape
        rgba_np = np.zeros((h, w, 4), dtype=np.uint8)


        yellow_mask = mask_np < 128
        rgba_np[yellow_mask] = [255, 204, 0, 128]
        rgba_np[~yellow_mask] = [0, 0, 0, 0]

        return Image.fromarray(rgba_np, 'RGBA')

    def draw_arrow(draw, x1, y1, x2, y2, color, width=5):
        """Draws a line and two arrowheads on the PIL ImageDraw object."""
        # Draw the main line


        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

        # Calculate angle of the line
        angle = math.atan2(y2 - y1, x2 - x1)
        head_len = 15  # Length of the arrowhead lines
        head_angle = math.pi / 6  # 30 degrees

        # --- Draw Head at (x2, y2) ---
        angle1 = angle + math.pi - head_angle
        angle2 = angle + math.pi + head_angle
        x2_h1 = x2 + head_len * math.cos(angle1)
        y2_h1 = y2 + head_len * math.sin(angle1)
        x2_h2 = x2 + head_len * math.cos(angle2)
        y2_h2 = y2 + head_len * math.sin(angle2)
        draw.line([(x2, y2), (x2_h1, y2_h1)], fill=color, width=width)
        draw.line([(x2, y2), (x2_h2, y2_h2)], fill=color, width=width)

        # --- Draw Head at (x1, y1) ---
        angle1 = angle - head_angle
        angle2 = angle + head_angle
        x1_h1 = x1 + head_len * math.cos(angle1)
        y1_h1 = y1 + head_len * math.sin(angle1)
        x1_h2 = x1 + head_len * math.cos(angle2)
        y1_h2 = y1 + head_len * math.sin(angle2)
        draw.line([(x1, y1), (x1_h1, y1_h1)], fill=color, width=width)
        draw.line([(x1, y1), (x1_h2, y1_h2)], fill=color, width=width)

    def colorize_layer(image_layer, target_color_rgb):

        layer_np = np.array(image_layer).copy()

        object_mask = layer_np[..., 3] > 0


        layer_np[object_mask, 0] = target_color_rgb[0]  # R
        layer_np[object_mask, 1] = target_color_rgb[1]  # G
        layer_np[object_mask, 2] = target_color_rgb[2]  # B

        return Image.fromarray(layer_np, "RGBA")


    def draw_labeled_arrow(object_img, label_img, x1, y1, x2, y2, label, color, color_text, width=5):

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


        draw_object = ImageDraw.Draw(object_img)
        draw_arrow(draw_object, x1, y1, x2, y2, color, width)

        if label:
            draw_label = ImageDraw.Draw(label_img)

            # ... (rest of the font/color setup remains the same) ...
            try:
                font = ImageFont.truetype("arial.ttf", size=20)
            except IOError:
                font = ImageFont.load_default()

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            angle = math.atan2(y2 - y1, x2 - x1)
            offset_distance = width * 2

            text_anchor_x = mid_x + offset_distance * math.cos(angle + math.pi / 2)
            text_anchor_y = mid_y + offset_distance * math.sin(angle + math.pi / 2)

            draw_label.text(  # Use draw_label here
                (text_anchor_x, text_anchor_y),
                label,
                fill=color_text,
                font=font,
                anchor="ma",
                stroke_width=1,
                stroke_fill=color_text
            )

    #rescales image
    TARGET_WIDTH = 1350
    # --- Base Image Scaling ---
    base_image_pil = None
    if st.session_state.original_image is not None:
        try:
            base_image_np = st.session_state.original_image
            rgb_array = cv2.cvtColor(base_image_np, cv2.COLOR_BGR2RGB)
            base_image_pil = Image.fromarray(rgb_array)

            new_height = math.floor(TARGET_WIDTH * base_image_pil.size[1] / base_image_pil.size[0])
            base_image_pil = base_image_pil.resize((TARGET_WIDTH, new_height))
        except Exception:
            # Fallback for processing error
            st.warning("Failed to process original image. Using a placeholder.")
    else:
        st.warning("No original image available. Using a placeholder.")

    base_image = base_image_pil.convert("RGBA")

    FINAL_IMAGE_SIZE = base_image.size  # (width, height)
    st.session_state.canvas_width = FINAL_IMAGE_SIZE[0]
    st.session_state.canvas_height = FINAL_IMAGE_SIZE[1]

    # --- Mask Image Scaling & Initialization (FIXED FOR NONE TYPE ERROR) ---

    mask_array_from_state = st.session_state.get("mask")
    mask_np_initial = None

    if mask_array_from_state is not None and isinstance(mask_array_from_state, np.ndarray):
        try:
            mask_image_pil = Image.fromarray(mask_array_from_state.astype(np.uint8), mode='L')

            if mask_image_pil.size != FINAL_IMAGE_SIZE:
                mask_image_pil = mask_image_pil.resize(FINAL_IMAGE_SIZE, Image.NEAREST)


            mask_np_initial = np.array(mask_image_pil)



        except Exception as e:
            st.error(f"Error preparing mask for editor: {e}")
            mask_np_initial = None

    if mask_np_initial is None:
        st.warning(
            "Mask is unavailable or failed processing. Using a completely transparent (erased) placeholder mask.")
        mask_np_initial = np.full((FINAL_IMAGE_SIZE[1], FINAL_IMAGE_SIZE[0]), 255, dtype=np.uint8)

    if "mask_np" not in st.session_state:
        st.session_state.mask_np = mask_np_initial

    if "temp_mask_np" not in st.session_state:
        st.session_state.temp_mask_np = st.session_state.mask_np.copy()

    if "canvas_version" not in st.session_state:
        st.session_state.canvas_version = 0

    if "object_layer_image" not in st.session_state:
        # This stores the accumulated objects (circles, lines, etc.)
        st.session_state.object_layer_image = Image.new("RGBA", FINAL_IMAGE_SIZE, (0, 0, 0, 0))
    if "arrow_layer_image" not in st.session_state:
        st.session_state.arrow_layer_image = Image.new("RGBA", FINAL_IMAGE_SIZE, (0, 0, 0, 0))

        # NEW: Layer for just the text labels
    if "label_layer_image" not in st.session_state:
        st.session_state.label_layer_image = Image.new("RGBA", FINAL_IMAGE_SIZE, (0, 0, 0, 0))
        st.session_state.arrow_label_input = ""
    if "mask_history" not in st.session_state:
        st.session_state.mask_history = []
    if "object_layer_history" not in st.session_state:
        # We need to store both the object and label layers together for 'Undo'
        st.session_state.object_layer_history = []
    if "label_layer_history" not in st.session_state:
        st.session_state.label_layer_history = []
    if "arrow_layer_history" not in st.session_state:
        st.session_state.arrow_layer_history = []
    if "interactive_objects" not in st.session_state:
        st.session_state.interactive_objects = []

    # NEW: Initialize pending_object for staged drawing
    if "pending_object" not in st.session_state:
        st.session_state.pending_object = None


    def mask_to_rgba(mask_np):

        if mask_np is None:
            return Image.new("RGBA", FINAL_IMAGE_SIZE, (0, 0, 0, 0))

        rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)

        rgba[mask_np < 128] = [255, 204, 0, 128]

        rgba[mask_np >= 128] = [0, 0, 0, 0]
        return Image.fromarray(rgba, "RGBA")


    def mask_to_black(mask_np):
        if mask_np is None:
            return Image.new("RGBA", FINAL_IMAGE_SIZE, (0, 0, 0, 0))
        rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        rgba[mask_np < 128] = [0, 0, 0, 0]
        rgba[mask_np >= 128] = [255, 255, 255, 255]
        return Image.fromarray(rgba, "RGBA")


    def render_car_to_image(car_json_list):

        car_img = Image.new("RGBA", FINAL_IMAGE_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(car_img)

        for obj in car_json_list:
            obj_type = obj.get("type")
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            points = obj.get("points", [])
            angle = obj.get("angle", 0)
            fill = obj.get("fill", "#FFFFFF")

            # Convert points relative to canvas
            xy = [(p['x'] + left, p['y'] + top) for p in points]

            if obj_type == "polygon":
                draw.polygon(xy, fill=fill, outline=obj.get("stroke", "black"))
            elif obj_type == "text":
                try:
                    font = ImageFont.truetype("arial.ttf", obj.get("fontSize", 20))
                except IOError:
                    font = ImageFont.load_default()
                # center alignment
                text_x = left
                text_y = top
                draw.text((text_x, text_y), obj["text"], font=font, fill=obj.get("fill", "black"))

        return car_img


    def create_car_json(cx, cy, length_m, width_m, scale, rotation, color):
        length_px = length_m * scale
        width_px = width_m * scale
        half_width = width_px / 2
        car_id = str(uuid.uuid4())

        body = {
            "type": "polygon",
            "points": [
                {"x": 0, "y": 0},
                {"x": 0, "y": width_px},
                {"x": length_px, "y": width_px},
                {"x": length_px, "y": 0},
            ],
            "left": cx - length_px / 2,
            "top": cy - half_width,
            "angle": rotation,
            "fill": color,
            "stroke": "black",
            "strokeWidth": 2,
            "id": car_id + "_body"
        }

        nose = {
            "type": "polygon",
            "points": [
                {"x": 0, "y": 0},
                {"x": 0, "y": width_px},
                {"x": length_px * 0.3, "y": half_width},
            ],
            "left": cx + length_px / 2 - (width_px * 3 / 4),
            "top": cy - half_width,
            "angle": rotation,
            "fill": "#FFD700",
            "stroke": "black",
            "strokeWidth": 2,
            "id": car_id + "_nose"
        }

        text = {
            "type": "text",
            "text": f"{length_m:.1f}m × {width_m:.1f}m",
            "left": cx,
            "top": cy - half_width - 25,
            "fontSize": max(10, int(scale / 2)),
            "fill": "black",
            "id": car_id + "_label"
        }

        return [body, nose, text]


    st.title("🖌️ Edit Lane Overlay Mask")
    st.write(
        "Pick an editing mode. You may make edits while app is reloading. Wait for app to finishing running (top right) before pressing 'Apply Change'. ")

    st.subheader("Controls")
    mode = st.radio("Mode", ["Paint", "Erase yellow", "Move Objects", "Insert Objects", "Distance Label"])

    #editing mode selection
    if mode == "Move Objects":
        drawing_tool = "Transform (Shapes)"
        fill_color = None
        brush_size = None
        stroke_color = None
        initial_drawing = None
    elif mode == "Distance Label":
        drawing_tool = "Line"
        stroke_color = "#ffcc00"
        fill_color = "rgba(255,204,0,0.5)"
        line_color = st.color_picker("Choose Line Color", value="#000000")
        text_color = st.color_picker("Choose text Color", value="#000000")
        brush_size = st.slider("Arrow size", 2, 50, 5)
        initial_drawing = None
        st.text_input("Label for next arrow:", key="arrow_label_input")
    elif mode == "Insert Objects":
        st.write("In order to move objects, change to 'Move Objects mode'")
        drawing_tool = st.radio("Objects", ["Line", "Circle", "Rectangle", "Polygon", "Car"])
        if drawing_tool == "Car":
            initial_drawing = {"objects": st.session_state.interactive_objects}
            drawing_tool = "Transform"
            st.sidebar.header("Car Parameters")
            car_x = 500
            car_y = 350
            rotation_deg = 0
            length_m = st.sidebar.slider("Car Length (m)", 2.0, 7.0, 4.5, 0.1)
            width_m = st.sidebar.slider("Car Width (m)", 1.5, 3.0, 1.8, 0.1)
            scale_px_m = st.session_state.scale_px_per_m
            car_color = st.sidebar.color_picker("Car Color", "#3B82F6")
            nose_length_factor = 0.3
            if st.sidebar.button("Add Car"):
                new_car = create_car_json(car_x, car_y, length_m, width_m, scale_px_m, rotation_deg, car_color)
                st.session_state.interactive_objects.extend(new_car)
                initial_drawing = {"objects": st.session_state.interactive_objects}
                st.session_state.canvas_version += 1
            if st.sidebar.button("Remove Car"):
                st.session_state.interactive_objects = []
                initial_drawing = {"objects": st.session_state.interactive_objects}
            fill_color = "rgba(255, 165, 0, 0.3)"
            brush_size = 2
            stroke_color = "#000000"
        elif drawing_tool == "Line":
            stroke_color = "#ffcc00"
            fill_color = "rgba(255,204,0,0.5)"
            brush_size = st.slider("Line size", 2, 50, 5)
            initial_drawing = None
        elif drawing_tool == "Rectangle":
            drawing_tool = "Rect"
            stroke_color = st.color_picker("Choose Shape Outline Color", value="#FF0000")
            brush_size = 5
            fill_color = None
            initial_drawing = None
        else:
            stroke_color = st.color_picker("Choose Shape Outline Color", value="#FF0000")
            brush_size = 5
            fill_color = None
            initial_drawing = None

    elif mode == "Paint":
        drawing_tool = "Freedraw"
        stroke_color = "#ffcc00"
        fill_color = "rgba(255,204,0,0.5)"
        brush_size = st.slider("Brush size", 5, 100, 10)
        initial_drawing = None

    elif mode == "Erase yellow":
        drawing_tool = "Freedraw"
        stroke_color = "#000000"
        fill_color = "rgba(255,204,0,0.5)"
        brush_size = st.slider("Brush size", 5, 100, 10)
        initial_drawing = None



    overlay_rgba = mask_to_rgba(st.session_state.temp_mask_np)
    intermediate_bg = Image.alpha_composite(base_image, overlay_rgba)
    composite_bg_objects = Image.alpha_composite(intermediate_bg, st.session_state.object_layer_image)
    intermediate_bg_2 = Image.alpha_composite(composite_bg_objects, st.session_state.arrow_layer_image)
    composite_bg = Image.alpha_composite(intermediate_bg_2, st.session_state.label_layer_image)
    #base image = original image
    #overlay stores road mask . paint and erase modes directly modify this.
    #object layer stores inserted objects, rectangle, car, line etc.
    #arrow layer stores the double-headed line of the distance label, label layer stores the distance label


    def save_state_for_undo():
        st.session_state.mask_history.append(st.session_state.temp_mask_np.copy())
        st.session_state.object_layer_history.append(st.session_state.object_layer_image.copy())
        st.session_state.arrow_layer_history.append(st.session_state.arrow_layer_image.copy())
        st.session_state.label_layer_history.append(st.session_state.label_layer_image.copy())

        if len(st.session_state.mask_history) > 20:
            st.session_state.mask_history.pop(0)
            st.session_state.object_layer_history.pop(0)
            st.session_state.arrow_layer_history.pop(0)
            st.session_state.label_layer_history.pop(0)


    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Apply Changes"):
            if st.session_state.pending_object is not None:
                save_state_for_undo()
                st.session_state.object_layer_image = Image.alpha_composite(
                    st.session_state.object_layer_image,
                    st.session_state.pending_object
                )
                st.session_state.pending_object = None
            st.session_state.interactive_objects = []
            st.session_state.mask_np = st.session_state.temp_mask_np.copy()
            st.success("Changes applied!")

            # Increment version to refresh the canvas
            st.session_state.canvas_version += 1
            st.rerun()

    with col2:
        if st.button("Undo"):
            if st.session_state.mask_history:
                # Restore the last saved state
                st.session_state.temp_mask_np = st.session_state.mask_history.pop()
                st.session_state.object_layer_image = st.session_state.object_layer_history.pop()
                st.session_state.arrow_layer_image = st.session_state.arrow_layer_history.pop()
                st.session_state.label_layer_image = st.session_state.label_layer_history.pop()

                st.session_state.canvas_version += 1
                st.warning("Undid last change.")
                st.rerun()
            else:
                st.info("Nothing to undo yet!")


    st.subheader("Canvas Editor")
    st.caption(
        "Draw here. 'Live Preview' below will show the proper colour. Background image will refresh after 'Apply Change' is pressed. If background image does not show, press apply changes. Press apply changes to commit edits before changing modes.")

    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=brush_size,
        stroke_color=stroke_color,
        background_image=composite_bg.convert("RGB"),
        initial_drawing=initial_drawing,
        update_streamlit=True,
        height=st.session_state.canvas_height,
        width=st.session_state.canvas_width,
        drawing_mode=drawing_tool.lower().split(" ")[0],
        key=f"mask_editor_canvas_{st.session_state.canvas_version}"
    )

    #code below handles registering the edits into the respective layers, but layers only update when Apply changes is pressed
    if canvas_result.json_data is not None and mode == "Distance Label": #replaces line with labelled two-headed arrow
        if canvas_result.json_data.get("objects"):
            new_object_index = len(canvas_result.json_data["objects"]) - 1
            new_object = canvas_result.json_data["objects"][new_object_index]

            if new_object["type"] == "line":
                center_x = new_object["left"]
                center_y = new_object["top"]
                relative_x1 = new_object["x1"]
                relative_y1 = new_object["y1"]
                relative_x2 = new_object["x2"]
                relative_y2 = new_object["y2"]

                x1 = center_x + relative_x1
                y1 = center_y + relative_y1
                x2 = center_x + relative_x2
                y2 = center_y + relative_y2

                save_state_for_undo()

                width = int(new_object.get("strokeWidth", 5))
                label = st.session_state.arrow_label_input

                draw_labeled_arrow(
                    st.session_state.arrow_layer_image,
                    st.session_state.label_layer_image,
                    x1, y1, x2, y2,
                    label, line_color,text_color, width
                )

                canvas_result.json_data["objects"].pop(new_object_index)

                st.session_state.canvas_version += 1
                st.rerun()

    elif canvas_result.image_data is not None:
        drawn_np = np.array(canvas_result.image_data).astype(np.uint8)
        drawn_alpha = drawn_np[..., 3]
        drawn_objects_only = drawn_np.copy()
        transparent_mask = drawn_alpha == 0
        drawn_objects_only[transparent_mask] = [0, 0, 0, 0]
        drawn_img = Image.fromarray(drawn_objects_only, "RGBA")

        if mode == "Insert Objects": #objects are saved in a pending layer for visualisation but changes are not committed
            st.session_state.pending_object = drawn_img

        elif mode == "Move Objects":
            st.session_state.pending_object = drawn_img
            pass

        elif mode == "Paint" or mode == "Erase yellow": #these directly edit the mask
            save_state_for_undo()
            drawn = np.array(canvas_result.image_data).astype(np.uint8)
            alpha = drawn[..., 3] > 10
            temp_mask = st.session_state.temp_mask_np.copy()
            if mode == "Paint":
                temp_mask[alpha] = 0
            else:  # Erase
                temp_mask[alpha] = 255
            st.session_state.temp_mask_np = temp_mask



    #Preview
    st.subheader("Live Preview")
    st.caption("Shows the base image + your *unsaved* changes.")

    preview_slot = st.empty()
    overlay_preview = mask_to_black(st.session_state.temp_mask_np)

    preview_composite = overlay_preview
    if st.session_state.pending_object is not None:
        preview_composite = Image.alpha_composite(preview_composite, st.session_state.pending_object)

    intermediate_1_preview = Image.alpha_composite(preview_composite, st.session_state.object_layer_image)
    intermediate_2_preview = Image.alpha_composite(intermediate_1_preview, st.session_state.arrow_layer_image)
    result_preview = Image.alpha_composite(intermediate_2_preview, st.session_state.label_layer_image)

    preview_slot.image(result_preview, use_column_width=True)

    if st.button("Save Sketch"):
        # Commit any pending objects before saving
        if st.session_state.pending_object is not None:
            st.session_state.object_layer_image = Image.alpha_composite(
                st.session_state.object_layer_image,
                st.session_state.pending_object
            )
            st.session_state.pending_object = None

        final_overlay = mask_to_black(st.session_state.temp_mask_np)
        final_1 = Image.alpha_composite(final_overlay, st.session_state.object_layer_image)
        final_2 = Image.alpha_composite(final_1, st.session_state.arrow_layer_image)
        final_result = Image.alpha_composite(final_2, st.session_state.label_layer_image)

        final_result.save("road_sketch_draft.png")
        st.session_state.road_sketch_draft = final_result
        st.success("Saved!")
        st.rerun()

if "road_sketch_draft" in st.session_state:
    st.image(st.session_state.road_sketch_draft)
