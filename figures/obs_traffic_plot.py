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
    "../deep_traffic_generation/data/training_datasets/landings_LFPO_06.pkl"
)

to = Traffic.from_file(
    "../deep_traffic_generation/data/training_datasets/takeoffs_LFPO_07.pkl"
)

# %%
with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 10), subplot_kw=dict(projection=EuroPP()), dpi=500
    )

    ldng[:2000].plot(ax, alpha=0.2, color="#ffbf79", zorder = 1)
    to[:2000].plot(ax, alpha=0.2, color="#9ecae9", zorder = 2)

    k1 = 2000 #east
    # k1 = 20 #west
    ldng[k1].plot(ax, color="#f58518", lw=1.5, zorder = 1)
    ldng[k1].at_ratio(0.2).plot(
        ax,
        color="#f58518",
        zorder=1,
        s=600,
        text_kw=dict(s=""))
    
    k2 = 10 #east
    # k2 = 20 #west
    to[k2].plot(ax, color="#4c78a8", lw=1.5, zorder = 2)
    to[k2].at_ratio(0.25).plot(
        ax,
        color="#4c78a8",
        zorder=2,
        s=600,
        text_kw=dict(s=""))
    
    navaids["MOLBA"].plot(ax, 
                          marker = 'p',
                          alpha = 0.7,
                          zorder = 5, 
                          shift=dict(units="dots", x=-300, y=0), 
                          text_kw=dict(
                            fontname="Fira Sans",
                            fontsize=10,))
    
    navaids["ODILO"].plot(ax, 
                          marker = 'p',
                          alpha = 0.7,
                          zorder = 5, 
                          shift=dict(units="dots", x=-280, y=0), 
                          text_kw=dict(
                            fontname="Fira Sans",
                            fontsize=10,))
    
    # navaids["VASOL"].plot(ax, 
    #                       marker = 'p',
    #                       alpha = 0.7,
    #                       zorder = 5, 
    #                       shift=dict(units="dots", x=-280, y=0), 
    #                       text_kw=dict(
    #                         fontname="Fira Sans",
    #                         fontsize=10,))
        
    # navaids["PO615"].plot(ax, 
    #                       marker = 'p',
    #                       alpha = 0.7,
    #                       zorder = 5, 
    #                       shift=dict(units="dots", x=20, y=60), 
    #                       text_kw=dict(
    #                         fontname="Fira Sans",
    #                         fontsize=10,))
    
    # navaids["PO300"].plot(ax, 
    #                       marker = '^',
    #                       alpha = 0.7,
    #                       zorder = 5, 
    #                       shift=dict(units="dots", x=30, y=0), 
    #                       text_kw=dict(
    #                         fontname="Fira Sans",
    #                         fontsize=10,))
    
    # navaids["PO301"].plot(ax,
    #                       marker = '^',
    #                       alpha = 0.7, 
    #                       zorder = 5, 
    #                       shift=dict(units="dots", x=-220, y=-120), 
    #                       text_kw=dict(
    #                         fontname="Fira Sans",
    #                         fontsize=10,))
    
    # navaids["PO303"].plot(ax,
    #                       marker = '^',
    #                       alpha = 0.7, 
    #                       zorder = 5, 
    #                       shift=dict(units="dots", x=30, y=60),
    #                       text_kw=dict(
    #                         fontname="Fira Sans",
    #                         fontsize=10,))
    
    # navaids["PO271"].plot(ax,
    #                       marker = '^',
    #                       alpha = 0.7, 
    #                       zorder = 5, 
    #                       shift=dict(units="dots", x=-250, y=-120), 
    #                       text_kw=dict(
    #                         fontname="Fira Sans",
    #                         fontsize=10,))

    airports["LFPO"].plot(ax, footprint=False, runways=dict(lw=1), labels=False)
    
    ax.set_title("East configuration", 
                 loc = "left",
                 y = 0.75,
                 fontsize=22, 
                 fontweight=570, 
                 fontstretch = 0)
    
    plt.margins(0,0)
    
    plt.show()

    fig.savefig("true_trajs_east.png", transparent=False, dpi=500)
    # fig.savefig("true_trajs_west.png", transparent=False, dpi=500)
# %%
