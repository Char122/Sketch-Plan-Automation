This is a road network implementation using osmnx package. osmnx is a network database so at the 50m scale the road geometries are wildy inaccurate.

The first section imports data from the API for a specified lat lon coordinate. 

The data is converted to metric coordinates so additional road features can be plotted. (gdf_edges and gdf_nodes)

Get width function tries different ways of getting road widths, first by trying to get it from the imported data and it falls back on the manual presets in the end.

Using this information, the rest of the code generates a simplified road using width + number of lanes to draw dashed lines parrelled to the road edges. It also labels the street names.


