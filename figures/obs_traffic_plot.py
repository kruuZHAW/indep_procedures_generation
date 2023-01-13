# %%
from traffic.core import Traffic
import matplotlib.pyplot as plt
from traffic.core.projection import EuroPP
from traffic.drawing import countries
from traffic.data import navaids
from traffic.data import airports
import numpy as np

# %%
ldng = Traffic.from_file(
    "../deep_traffic_generation/data/training_datasets/landings_LFPO_25.pkl"
)

to = Traffic.from_file(
    "../deep_traffic_generation/data/training_datasets/takeoffs_LFPO_24.pkl"
)

# %%
with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 10), subplot_kw=dict(projection=EuroPP()), dpi=500
    )

    ldng[:2000].plot(ax, alpha=0.2, color="#ffbf79")
    to[:2000].plot(ax, alpha=0.2, color="#9ecae9")

    # k1 = 2000 #east
    k1 = 20 #west
    ldng[k1].plot(ax, color="#f58518", lw=1.5)
    ldng[k1].at_ratio(0.1).plot(
        ax,
        color="#f58518",
        zorder=3,
        s=600,
        shift=dict(units="dots", x=500, y=80),
        text_kw=dict(
            fontname="Fira Sans",
            fontsize=12,
            ha="right",
            bbox=dict(
                boxstyle="round",
                edgecolor="none",
                facecolor="white",
                alpha=0.7,
                zorder=5,
            ),
        ),
    )
    
    # k2 = 10 #east
    k2 = 20 #west
    to[k2].plot(ax, color="#4c78a8", lw=1.5)
    to[k2].at_ratio(0.3).plot(
        ax,
        color="#4c78a8",
        zorder=3,
        s=600,
        shift=dict(units="dots", x=600, y=60),
        text_kw=dict(
            fontname="Fira Sans",
            fontsize=12,
            ha="right",
            bbox=dict(
                boxstyle="round",
                edgecolor="none",
                facecolor="white",
                alpha=0.7,
                zorder=5,
            ),
        ),
    )

    airports["LFPO"].plot(ax, footprint=False, runways=dict(lw=1), labels=False)
    plt.show()

    # fig.savefig("true_trajs_east.png", transparent=False, dpi=500)
    fig.savefig("true_trajs_west.png", transparent=False, dpi=500)
# %%
