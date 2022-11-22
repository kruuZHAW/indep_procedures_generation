# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:04:29 2022

@author: sarah, alessio, adrian
"""
 
# %%
## Modules
from traffic.data import opensky, airports, eurofirs, navaids
from cartes.crs import EuroPP, LambertConformal
from cartes.utils.features import countries, lakes, ocean
import matplotlib.pyplot as plt
import pandas as pd
from traffic.data import opensky
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm
from traffic.core import Traffic
import pandas as pd
from statistics import mean
import numpy
from datetime import timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Funktionen
# %% 
def turb(name):   

    #def
    fin_data = pd.DataFrame([])
    ids = []

    ## import data
    t = Traffic.from_file(name)

    ## assign flight id
    t = t.assign_id().eval()

    ## neglect flights with less than 5 minutes duration
    for f in t:
        if f.duration > timedelta(minutes = 5):
            ids.append(f.flight_id)
            
    t[ids]

    ## filter data with no v/s values
    t.data.dropna(subset=['vertical_rate_barometric', 'vertical_rate_inertial'])
    t_processed2 = t.filter(vertical_rate_barometric = 3, vertical_rate_inertial = 3, strategy = None).agg_time("30s", vertical_rate_barometric ="std", vertical_rate_inertial = "std", latitude="mean", longitude = "mean").eval(desc = "")

    ## Umwandlung zu DataFrame, Speichern der Rohdaten & y2 rechnen
    dfdata = pd.DataFrame(t_processed2.data)
    dfdata["y2"] = abs(dfdata.vertical_rate_barometric_std - dfdata.vertical_rate_inertial_std)

    # Get all unique Flight IDs
    flightID = dfdata.flight_id.unique()

    for i in flightID:
         selected_flight = dfdata.loc[dfdata['flight_id'] == i]
         threshold = np.mean(selected_flight.y2) + 1.2 * np.std(selected_flight.y2)
         selected_flight["Turb"] = selected_flight.y2 > threshold
         fin_data = fin_data.append(selected_flight)

    return fin_data[["timestamp","callsign","latitude","longitude","flight_id","y2","Turb"]]
            
def getdata(start_day, end_day, name):
    ## enter all parameters needed
    from shapely.geometry import Polygon
    shape = Polygon([[5,43],[5,43],[9,46],[9,46]])

    data = opensky.history(
        start=start_day,
        stop=end_day,
        cached=False,
        bounds=shape,
        other_params=" and geoaltitude >9000 " ##altitude in METERS
        
    )

    df = opensky.extended(
        start=start_day,
        stop=end_day,
        bounds=shape,
        cached=False,
    )

    # save the data to a csv, apply filters

    extended_data = data.filter().query_ehs(df).resample('1s').eval()

    t = extended_data.assign_id().eval()

    t2 = t.data.dropna(subset=['vertical_rate_barometric', 'vertical_rate_inertial'])
    
    t2.to_pickle(name)
    return 

# %%
# Inp
## shape muss in def get data eingegeben werden
## shape muss beim plotten unten bei set_extent gesetzt werden (Rahmen +.05° zu den Koordinaten)
start = "2021-07-25 18:00"
end = "2021-07-25 19:00"
dateiname = "RQ2_1_data.pkl"


#function
getdata(start, end, dateiname)   #Auskommentieren wenn bereits einmal ausgeführt


alles = turb(dateiname)


## specify colors
colors = np.where(alles["Turb"]==True,'r','silver')

# %%
# Create the map
fig = plt.figure(figsize=(15,10))

ca_map = plt.axes(projection=LambertConformal(7, 44))
ca_map.set_extent((5, 9, 43, 46))  #set coordinates for visualization (°E °E, °N °N)

ca_map.add_feature(countries(scale="50m"))
ca_map.add_feature(lakes(scale="50m"))
ca_map.add_feature(ocean(scale="50m"))
ca_map.set_title("Airspace above Nice (FL280+, 25-07-2021)")
# ca_map.xaxis()
# ca_map.set_xlabel("Longitude")
# ca_map.set_ylabel("Latitude")
ca_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)

airports["LFLL"].point.plot(ca_map, text_kw=dict(s="  Lyon Airport", fontsize=14))
airports["LFML"].point.plot(ca_map, text_kw=dict(s="  Marseille Airport", fontsize=14))
airports["LFMN"].point.plot(ca_map, text_kw=dict(s="  Nice Airport, FR", fontsize=14))
airports["LIMF"].point.plot(ca_map, text_kw=dict(s="  Turin Airport, IT", fontsize=14))

# Plot the data onto map
ca_map.scatter(alles['longitude'],
            alles['latitude'],  
            c=colors,
            s = 5,
            transform=ccrs.PlateCarree(),
            )


# %%
