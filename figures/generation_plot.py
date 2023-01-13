# %%
import matplotlib.pyplot as plt
import pandas as pd
from traffic.core import Traffic
from os import walk

# %%
# gen_to = Traffic.from_file("generated_to_east.pkl")
# gen_ldng = Traffic.from_file("generated_ldng_east.pkl")

gen_to = Traffic.from_file("generated_to_west.pkl")
gen_ldng = Traffic.from_file("generated_ldng_west.pkl")

# %%
gen_to = gen_to.iterate_lazy().cumulative_distance().eval(max_workers = 50)
gen_ldng = gen_ldng.iterate_lazy().cumulative_distance(reverse = True).eval(max_workers = 50)

# %%
from traffic.core.projection import EuroPP, PlateCarree
from traffic.data import airports
import numpy as np
import matplotlib.patches as mpatches

#East
# x = np.array([2.07, 2.07, 2.5, 2.5])
# y = np.array([48.6, 48.3, 48.3, 48.6])

#West
x = np.array([2.0, 2.0, 2.3, 2.3])
y = np.array([48.7, 48.4, 48.4, 48.7])

poly_corners = np.zeros((len(y), 2), np.float64)
poly_corners[:,0] = x
poly_corners[:,1] = y
poly1 = mpatches.Polygon(poly_corners, 
                        closed=True, 
                        alpha = 0.1,
                        ec='#b22222', 
                        fill=True, 
                        fc='#b22222', 
                        transform = PlateCarree())

poly2 = mpatches.Polygon(poly_corners, 
                        closed=True, 
                        alpha = 1,
                        ec='#b22222', 
                        fill=False, 
                        lw=5,
                        linestyle = "--",
                        transform = PlateCarree())

with plt.style.context("traffic"):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection=PlateCarree())

    ax.set_title("Generated synthetic trajectories", pad=0, fontsize=20)

    gen_to.plot(ax, alpha=0.3, color="#9ecae9")
    gen_to["TRAJ_0"].plot(ax, color="#4c78a8", lw=2)
    gen_to["TRAJ_0"].at_ratio(0.3).plot(
        ax,
        color="#4c78a8",
        zorder=5,
        s=600,
        text_kw={"s": None},
    )

    gen_ldng.plot(ax, alpha=0.3, color="#ffbf79")
    gen_ldng["TRAJ_0"].plot(ax, color="#f58518", lw=2)
    gen_ldng["TRAJ_0"].at_ratio(0.4).plot(
        ax,
        color="#f58518",
        zorder=5,
        s=600,
        text_kw={"s": None},
    )
    airports["LFPO"].point.plot(ax, shift=dict(units="dots", x=30, y=-60),text_kw=dict(fontname="Fira Sans", fontsize=15))
    
    ax.add_patch(poly1)
    ax.add_patch(poly2)
    plt.show()

# fig.savefig("generation_east.png", transparent=False, dpi=300)
fig.savefig("generation_west.png", transparent=False, dpi=300)

# %%
#Get cumdist coordinates for the box

#East
# test_to = gen_to.query("(48.3 < latitude < 48.6) & (2.07 < longitude < 2.5)")
# test_ldng = gen_ldng.query("(48.3 < latitude < 48.6) & (2.07 < longitude < 2.5)")

#West
test_to = gen_to.query("(48.4 < latitude < 48.7) & (2.0 < longitude < 2.3)")
test_ldng = gen_ldng.query("(48.4 < latitude < 48.7) & (2.0 < longitude < 2.3)")

print(test_to.data.cumdist.min(), test_to.data.cumdist.max())
print(test_ldng.data.cumdist.min(), test_ldng.data.cumdist.max())

# %%
import altair as alt

# Just put the pseudo-input at the end for display
copy_traf_1 = gen_to
a = copy_traf_1["TRAJ_0"].assign(flight_id="TRAJ_999")
copy_traf_1 = copy_traf_1 + a

copy_traf_2 = gen_ldng
b = copy_traf_2["TRAJ_0"].assign(flight_id="TRAJ_999")
copy_traf_2 = copy_traf_2 + b

# lines1 = alt.Chart(pd.DataFrame({'cumdist': [15.25, 41.21]}) #east
lines1 = alt.Chart(pd.DataFrame({'cumdist': [0.3, 24.9]}) #west
                   ).mark_rule().encode(x="cumdist", 
                                    color = alt.value('#b22222'),
                                    strokeWidth = alt.value(4),
                                    strokeDash = alt.value([5,5]))

# lines2 = alt.Chart(pd.DataFrame({'cumdist': [21, 48.25]}) #east
lines2 = alt.Chart(pd.DataFrame({'cumdist': [37.4, 58.5]}) #east
                   ).mark_rule().encode(x="cumdist", 
                                    color = alt.value('#b22222'),
                                    strokeWidth = alt.value(4),
                                    strokeDash = alt.value([5,5]))

# area1 = alt.Chart(pd.DataFrame({'x1': [15.25], 'x2': [41.21]}) #east
area1 = alt.Chart(pd.DataFrame({'x1': [0.3], 'x2': [24.9]}) #west
).mark_rect(
    opacity=0.1
).encode(
    x="x1",
    x2="x2",
    y=alt.value(0),  # pixels from top
    y2=alt.value(300),  # pixels from top
    color=alt.value('#b22222')
)

# area2 = alt.Chart(pd.DataFrame({'x1': [21], 'x2': [48.25]}) #east
area2 = alt.Chart(pd.DataFrame({'x1': [37.4], 'x2': [58.5]}) #west
).mark_rect(
    opacity=0.1
).encode(
    x="x1",
    x2="x2",
    y=alt.value(0),  # pixels from top
    y2=alt.value(300),  # pixels from top
    color=alt.value('#b22222')
)

chart1 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "cumdist",
                title="Distance from start (in Nm)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value(1),
                alt.value(0.2),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value("#4c78a8"),
                alt.value("#9ecae9"),
            ),
        )
        for flight in copy_traf_1
    )
).properties(title="Departure altitude (in ft)")

chart2 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "cumdist",
                title="Distance from start (in Nm)",
            ),
            y=alt.Y("groundspeed", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value(1),
                alt.value(0.2),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value("#4c78a8"),
                alt.value("#9ecae9"),
            ),
        )
        for flight in copy_traf_1
    )
).properties(title="Departure ground speed (in kts)")

chart3 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "cumdist",
                scale=alt.Scale(reverse=True),
                title="Distance till end (in Nm)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value(1),
                alt.value(0.2),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value("#f58518"),
                alt.value("#ffbf79"),
            ),
        )
        for flight in copy_traf_2
    )
).properties(title="Arrival altitude (in ft)")

chart4 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "cumdist",
                scale=alt.Scale(reverse=True),
                title="Distance till end (in Nm)",
            ),
            y=alt.Y("groundspeed", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value(1),
                alt.value(0.2),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value("#f58518"),
                alt.value("#ffbf79"),
            ),
        )
        for flight in copy_traf_2
    )
).properties(title="Arrival ground speed (in kts)")

plots = (
    alt.vconcat(alt.hconcat(chart1+lines1+area1, chart2+lines1+area1), alt.hconcat(chart3+lines2+area2, chart4+lines2+area2))
    .configure_title(fontSize=22)
    .configure_axis(labelFontSize=16, titleFontSize=19)
)

plots

# %%
