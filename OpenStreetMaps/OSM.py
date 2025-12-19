import osmnx as ox
import matplotlib.pyplot as plt
import os
import math

#fetching data from API
ox.settings.overpass_url = 'https://overpass-api.de/api/interpreter'
plt.switch_backend('Agg')

latitude, longitude = 1.3516639, 103.9432205
point = (latitude, longitude)
buffer_radius = 400

G = ox.graph_from_point(point, dist=buffer_radius, dist_type='bbox', network_type='drive')
tags = {'highway': ['traffic_signals', 'street_lamp'], 'amenity': ['traffic_signals']}
place_gdf = ox.features_from_point(point, tags, dist=buffer_radius)
#metric coordinates
G_projected = ox.project_graph(G)
gdf_edges = ox.graph_to_gdfs(G_projected, nodes=False, edges=True)
gdf_nodes = ox.graph_to_gdfs(G_projected, nodes=True, edges=False)

#presets road widths for different lane types
standard_lane_width = 3.5
default_widths = {
    'motorway': 6 * standard_lane_width, 'trunk': 4 * standard_lane_width,
    'primary': 3 * standard_lane_width, 'secondary': 2 * standard_lane_width,
    'tertiary': 1.5 * standard_lane_width, 'residential': 1.5 * standard_lane_width,
    'service': 1.0 * standard_lane_width, 'unclassified': 1.5 * standard_lane_width,
    'default': 1.5 * standard_lane_width
}

#tries different ways to get road width
def get_width(row, G_projected):
    u, v, key = row.name 
    width, lanes, highway = row.get('width'), row.get('lanes'), row.get('highway')

    if lanes:
        if isinstance(lanes, list): lanes = lanes[0]
        try:
            lanes = float(lanes.split('-')[1]) if '-' in str(lanes) else float(lanes)
            return lanes * standard_lane_width
        except (ValueError, TypeError):
            pass

    if width:
        return float(width[0]) if isinstance(width, list) else float(width)

    try:
        for neighbor_node, neighbor_edges in G_projected[u].items():
            for _, neighbor_data in neighbor_edges.items():
                neighbor_lanes = neighbor_data.get('lanes')
                neighbor_width = neighbor_data.get('width')
                if neighbor_lanes or neighbor_width:
                    if neighbor_lanes:
                        if isinstance(neighbor_lanes, list): neighbor_lanes = neighbor_lanes[0]
                        try:
                            return float(neighbor_lanes) * standard_lane_width
                        except (ValueError, TypeError):
                            pass
                    if neighbor_width:
                        return float(neighbor_width)
        for neighbor_node, neighbor_edges in G_projected[v].items():
            for _, neighbor_data in neighbor_edges.items():
                neighbor_lanes = neighbor_data.get('lanes')
                neighbor_width = neighbor_data.get('width')
                if neighbor_lanes or neighbor_width:
                    if neighbor_lanes:
                        if isinstance(neighbor_lanes, list): neighbor_lanes = neighbor_lanes[0]
                        try:
                            return float(neighbor_lanes) * standard_lane_width
                        except (ValueError, TypeError):
                            pass
                    if neighbor_width:
                        return float(neighbor_width)
    except Exception as e:
        print(f"Could not infer from neighbors for edge ({u}, {v}): {e}")

    if isinstance(highway, list): highway = highway[0]
    return default_widths.get(highway, default_widths['default'])


gdf_edges['road_width'] = gdf_edges.apply(lambda row: get_width(row, G_projected), axis=1)

traffic_lights_gdf = place_gdf[place_gdf['highway'].isin(['traffic_signals'])]
street_lamps_gdf = place_gdf[place_gdf['highway'].isin(['street_lamp'])]

gdf_edges_polygons = gdf_edges.copy()
gdf_edges_polygons['geometry'] = gdf_edges_polygons.geometry.buffer(gdf_edges_polygons['road_width'] / 2,
                                                                    cap_style=2)

road_colors = {
    'motorway': '#7F0000', 'trunk': '#FF5733', 'primary': '#003366',
    'secondary': '#336633', 'tertiary': '#666666', 'residential': '#999999',
    'service': '#AAAAAA', 'unclassified': '#CCCCCC', 'default': '#999999'
}
#plots roads
fig, ax = ox.plot_graph(G_projected, bgcolor='white', node_size=0, show=False, close=True)

gdf_edges_polygons.plot(ax=ax, color=gdf_edges_polygons['highway'].map(
    lambda x: road_colors.get(x[0] if isinstance(x, list) else x, road_colors['default'])), zorder=1)

for _, row in gdf_edges.iterrows():
    if row.geometry is not None and not math.isnan(row['road_width']):
        num_lanes = int(round(row['road_width'] / standard_lane_width))
        if num_lanes > 1:
            mid_offset = (standard_lane_width / 2)
            for i in range(1, num_lanes):
                line = row.geometry
                if line.geom_type == 'LineString':
                    lane_offset = mid_offset * (2 * i - num_lanes)
                    try:
                        offset_line = line.parallel_offset(lane_offset, 'left')
                        ax.plot(*offset_line.xy, color='#FFFFFF', linestyle='--', linewidth=0.75, zorder=3)
                    except Exception:
                        pass
#labels streets
named_streets = gdf_edges.dropna(subset=['name']).groupby('name')
label_y_offset = 0
text_x_pos = max(gdf_edges.geometry.bounds['maxx']) + 50

for name, group in named_streets:
    street_name = name[0] if isinstance(name, list) else name
    longest_segment = group.loc[group['length'].idxmax()]

    if longest_segment['length'] > 20:
        if longest_segment.geometry is not None and longest_segment.geometry.geom_type == 'LineString':
            line = longest_segment.geometry
            midpoint = line.centroid
            total_length = round(group['length'].sum(), 2)
            combined_label = f"{street_name} ({total_length}m)"

            ax.annotate(combined_label,
                        xy=midpoint.coords[0],
                        xytext=(text_x_pos, midpoint.y + label_y_offset),
                        arrowprops=dict(facecolor='black', arrowstyle='-', shrinkB=0, linewidth=1),
                        fontsize=8, ha='left', va='center', color='black', zorder=5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

            label_y_offset += 25
#data seems to not exist
if not traffic_lights_gdf.empty:
    traffic_lights_gdf.plot(ax=ax, marker='s', color='red', markersize=30, label='Traffic Light', zorder=6)

if not street_lamps_gdf.empty:
    street_lamps_gdf.plot(ax=ax, marker='o', color='yellow', markersize=10, label='Street Lamp', zorder=6)

plt.title("Detailed 2D Road Sketch", fontsize=14)
plt.legend(loc='lower right', framealpha=0.8)
plt.show()

output_filename = 'detailed_2d_road_sketch.png'
fig.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {os.path.abspath(output_filename)}")
