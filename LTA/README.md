There's two versions, one generates a .png and the other .dxf. The .dxf can be imported into FARO Zone. The .png version only works at 50m radius as the arrows are sized for that. You could resize them dynamically with the radius, but the code's not there at the moment.

For the .dxf version, the first section of the code is just some tunable parameters. The only ones that the users need to input are radius and lat-lon coords. There are some hard coded dictionaries since each API has its own data structure. 

get_feature_type() and get_feature_name() deal with data extraction.

get_filtered_cache_key(), load_filtered_features() and save_filtered_features() do caching and cache loading.

get_geojson_data() is the LTA example query written into a function.

get_first_coord(), extract_all_coords_from_feature() and filter_features_by_radius() work together. The code first reads only the first coordinate of the object via get_first_coord(). If it is within the requested radius + 150m, we will check the rest of its coordinates via extract_all_coords_from_feature() to see if any falls within the radius. If it does, it is saved. For kerbs, this is set to radius + 1000m since kerbs can be very large and the arbitrary one point we pick may be far away even if the kerb actually is within the radius. This is a optimisation to avoid excesive computation.

convert_wgs_to_local_xy() converts lat-lon to meters.

setup_dxf_linetypes(), create_dxf_layers() and setup_arrow_blocks() are helper functions to the export function. setup_arrow_blocks() calls .dxf files inside a custom arrows image folder (should be uploaded in the github) and creates a block for it which can then be inserted into the exported .dxf. Kind of like matplotlib inserting pngs.

export_road_features_to_dxf() does most of the work. There are draw functions inside for each road feature.

main file bascially runs all the functions above linearly. process_dataset() is a helper function for caching step. 


