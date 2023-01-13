# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm 
import altair as alt

from traffic.core import Traffic

# %%
t1 = Traffic.from_file("../deep_traffic_generation/data/training_datasets/landings_LFPO_25.pkl")
t1_gen = Traffic.from_file("../deep_traffic_generation/data/generated_datasets/gen_ldng_LFPO_25_1.pkl")

t2 = Traffic.from_file("../deep_traffic_generation/data/training_datasets/takeoffs_LFPO_24.pkl")
t2_gen = Traffic.from_file("../deep_traffic_generation/data/generated_datasets/gen_to_LFPO_24_1.pkl")

# %%
t1 = t1.iterate_lazy().cumulative_distance(reverse = True).eval(desc = "", max_workers = 50)
t1_gen_short = t1_gen[:1000].iterate_lazy().cumulative_distance(reverse = True).eval(desc = "", max_workers = 50)

t2 = t2.iterate_lazy().cumulative_distance().eval(desc = "", max_workers = 50)
t2_gen_short = t2_gen[:1000].iterate_lazy().cumulative_distance().eval(desc = "", max_workers = 50)

# %%

chart1 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "cumdist",
                scale=alt.Scale(reverse=True),
                title="Distance from start (in Nm)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.value(0.2),
            color=alt.value("#9ecae9"),
        )
        for flight in t1[:500]
    )
).properties(
    width=800,
    height=300
)

chart2 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "cumdist",
                scale=alt.Scale(reverse=True),
                title="Distance from start (in Nm)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.value(0.2),
            color=alt.value("#ffbf79"),
        )
        for flight in t1_gen_short[:500]
    )
).properties(
    width=800,
    height=800,
)

chart2 + chart1

# %%
chart3 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "cumdist",
                title="Distance from start (in Nm)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.value(0.2),
            color=alt.value("#9ecae9"),
        )
        for flight in t2[:500]
    )
).properties(
    width=800,
    height=300
)

chart4 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "cumdist",
                title="Distance from start (in Nm)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.value(0.2),
            color=alt.value("#ffbf79"),
        )
        for flight in t2_gen_short[:500]
    )
).properties(
    width=800,
    height=300,
)

chart4+chart3
# %%
