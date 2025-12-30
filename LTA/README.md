# LTA Database 
## Overview
The LTA Database is a collection of APIs that record various aspects of road elements in Singapore. The data has various formats and hence the code that queries it needs to be customised between APIs.

There's two versions, one generates a .png and the other .dxf. The .dxf can be imported into FARO Zone. The .png version only works at 50m radius as the arrows are sized for that. You could resize them dynamically with the radius, but the code's not there at the moment. The .png version also does not include traffic lights or lamp posts, but that can be added.

## LTA-DXF
The first section of the code is just some tunable parameters. The only ones that the users need to input are radius and lat-lon coords. There are some hard coded dictionaries since each API has its own data structure. 

### Data Extraction
* get_feature_type() - extracts road feature type.
* get_feature_name() - extracts names of road feature.

### Caching
* get_filtered_cache_key() - generates a cache key via hashing.
* load_filtered_features() - loads cache using key.
* save_filtered_features() - saves cache using key.

### Query
get_geojson_data() - the LTA example query written into a function.

### Data Filtering
* get_first_coord() - reads the first coordinate of the object. Helper function.
* extract_all_coords_from_feature() - gets all the coordinates from the object. Helper function.
* filter_features_by_radius() - First checks the first coordinate of the object via get_first_coord(). If radius + 150m (+1000m for kerbs), checks the rest of the coordinates via extract_all_coords_from_feature(). If any falls within the radius, saves object. This is a optimisation to avoid excesive computation.

### Lat-Lon to meters
convert_wgs_to_local_xy()  - converts lat-lon to meters.

### DXF Export
* setup_dxf_linetypes() - sets up line formats. After being imported, FARO Zone 3D seems to revert back to default line settings.
* create_dxf_layers() - creates layers for the different road elements.
* setup_arrow_blocks() - calls .dxf files inside a custom arrows image folder (should be uploaded in the github) and creates a block for it which can then be inserted into the exported .dxf.
* export_road_features_to_dxf() - has draws functions for each road element. Uses the helper functions above. 

### Main
The main block bascially runs all the functions above linearly. process_dataset() is a helper function for caching step. 

## LTA-PNG
### Repeated Functions
get_feature_type(), get_feature_name(), get_filtered_cache_key(), load_filtered_features(), save_filtered_features(), get_geojson_data(), get_first_coord(), extract_all_coords_from_feature() and filter_features_by_radius() all have similiar functionalities as above.

### Custom Arrows
preload_arrow_images() preloads arrow images from the custom_arrows_images folder, this time in .png format.

### PNG Plotting
draw_road_features() is similiar to export_road_features_to_dxf() but instead uses matplotlib

### Main
the main block functions the same as above.

