#%%
from shapely.geometry import Point
from tqdm.autonotebook import tqdm 

from traffic.core import Traffic
from traffic.data import airports
from ipyleaflet import Map, basemaps, Marker, MarkerCluster, Polygon, AwesomeIcon
from ipywidgets import Layout

#%%
obs_to_east = Traffic.from_file("../deep_traffic_generation/data/training_datasets/takeoffs_LFPO_07.pkl")
obs_ldng_east = Traffic.from_file("../deep_traffic_generation/data/training_datasets/landings_LFPO_06.pkl")

obs_to_west = Traffic.from_file("../deep_traffic_generation/data/training_datasets/takeoffs_LFPO_24.pkl")
obs_ldng_west = Traffic.from_file("../deep_traffic_generation/data/training_datasets/landings_LFPO_25.pkl")

#%%
last_track = obs_to_east.data.groupby("flight_id")["track"].last()
id_to_south = last_track[(last_track > 130) & (last_track < 210)].index
obs_to_east = obs_to_east[id_to_south]
s_obs_to_east = obs_to_east[:1000]

first_track = obs_ldng_east.data.groupby("flight_id")["track"].first()
id_from_south = first_track[first_track > 270].index
obs_ldng_east = obs_ldng_east[id_from_south]
s_obs_ldng_east = obs_ldng_east[:1000]

intersect_east = []

for f1, f2 in tqdm(zip(s_obs_to_east, s_obs_ldng_east)):
    inter = f1.shape.intersection(f2.shape)
    if isinstance(inter, Point):
        intersect_east.append({"lat": inter.y, "lon":inter.x})
        
#%%
last_track = obs_to_west.data.groupby("flight_id")["track"].last()
id_to_south = last_track[(last_track > 45) & (last_track < 210)].index
obs_to_west = obs_to_west[id_to_south]
s_obs_to_west = obs_to_west[:1000]

first_track = obs_ldng_west.data.groupby("flight_id")["track"].first()
id_from_south = first_track[(first_track > 15) & (first_track < 150)].index
obs_ldng_west = obs_ldng_west[id_from_south]
s_obs_ldng_west = obs_ldng_west[:1000]

intersect_west = []

for f1, f2 in tqdm(zip(s_obs_to_west, s_obs_ldng_west)):
    inter = f1.shape.intersection(f2.shape)
    if isinstance(inter, Point):
        intersect_west.append({"lat": inter.y, "lon":inter.x})
        
#%%

map_ = Map(
    center=airports["LFPO"].latlon,
    zoom=10,
    basemap=basemaps.Stamen.Terrain,
    layout=Layout(width="100%", height="1000px"),
)

icon1 = AwesomeIcon(
    name='plane',
    marker_color='red',
    icon_color='white',
    spin=False
)
marker1 = Marker(icon=icon1, location=airports["LFPO"].latlon, draggable=False)
map_.add_layer(marker1)

icon2 = AwesomeIcon(
    # name='ambulance',
    name='exclamation-triangle',
    marker_color='blue',
    icon_color='white',
    spin=False
)

icon3 = AwesomeIcon(
    # name='ambulance',
    name='exclamation-triangle',
    marker_color='orange',
    icon_color='white',
    spin=False
)

# markers = []
for l in intersect_west:
    # markers.append(Marker(icon = icon2, location=(l["lat"], l["lon"]), draggable=False)) 
    map_.add_layer(Marker(icon = icon2, location=(l["lat"], l["lon"]), draggable=False, opacity = 0.5))
    
for l in intersect_east:
    # markers.append(Marker(icon = icon2, location=(l["lat"], l["lon"]), draggable=False)) 
    map_.add_layer(Marker(icon = icon3, location=(l["lat"], l["lon"]), draggable=False, opacity = 0.5))

# marker_cluster = MarkerCluster(
#     markers=markers
# )
# map_.add_layer(marker_cluster)

polygon1 = Polygon(
    locations=[(48.70, 2.0), (48.30, 2.0), (48.30, 2.70), (48.70,2.70)],
    color="#4c78a8",
    fill_color="#4c78a8"
)
map_.add_layer(polygon1)

polygon2 = Polygon(
    locations=[(48.6102599, 2.163190), (48.267231, 2.163190), (48.267231,2.82), (48.6102599,2.82)],
    color="#f58518",
    fill_color="#f58518"
)
map_.add_layer(polygon2)

map_
# %%
