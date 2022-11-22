# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:51:59 2022

@author: sarah
"""

from traffic.data import opensky
from cartes.crs import EuroPP
from cartes.utils.features import countries
import matplotlib.pyplot as plt
import pandas as pd


## enter all parameters needed
start_day = "2022-10-18 09:30"
end_day = "2022-10-18 11:30"


from shapely.geometry import Polygon
shape = Polygon([[-1,52],[-1,53],[2,53],[2,52]])


data = opensky.history(
    start=start_day,
    stop=end_day,
    cached=False,
    bounds=shape,
    other_params=" and geoaltitude >3000 " ##altitude in METERS
)

df = opensky.extended(
    start=start_day,
    stop=end_day,
    bounds=shape,
    cached=False,
)


# save the data to a parquet, apply filters

extended_data = data.filter().query_ehs(df).resample('1s').eval(max_workers=50, desc = "")

extended_data.to_pickle("RQ2_data.pkl")


