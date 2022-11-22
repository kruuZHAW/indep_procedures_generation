from traffic.data import opensky
from shapely.geometry import Polygon
shape = Polygon([[7, 47], [7, 48], [8, 48], [8, 47]])

schiphol = opensky.history(
    "2022-08-19 12:00",
    "2022-08-19 14:00",
    # airport="LSZH"
    bounds=shape
)

df = opensky.extended(
    "2022-08-19 12:00",
    "2022-08-19 14:00",
    # airport="LSZH"
    bounds=shape
)

enriched_lszh = (
    schiphol
    .filter()
    .query_ehs(df)
    .resample('1s')
    .eval(desc='', max_workers=50)
)

# enriched_lszh.to_parquet("test_mode_s_dataset.parquet.gz")
enriched_lszh.to_pickle("RQ2_data.pkl")