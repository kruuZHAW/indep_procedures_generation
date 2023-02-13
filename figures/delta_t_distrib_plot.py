#%%
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm 
from traffic.core import Traffic

#%%
dist = st.fatiguelife(c = 1.1635939736232785, loc = 52.634030248193504, scale = 251.13235704363382)
dist.mean()

#%%
obs_ldng = Traffic.from_file("../deep_traffic_generation/data/training_datasets/landings_LFPO_06.pkl")
first_track = obs_ldng.data.groupby("flight_id")["track"].first()
id_from_south = first_track[first_track > 270].index
obs_ldng = obs_ldng[id_from_south]

#%%
end_time = obs_ldng.data.groupby("flight_id")["timestamp"].last().sort_values()
df_end_time = end_time.to_frame()
df_end_time["year"] = end_time.dt.year
df_end_time["month"] = end_time.dt.month
df_end_time["day"] = end_time.dt.day
df_end_time["hour"] = end_time.dt.hour
df_end_time["day_name"] = end_time.dt.day_name()
df_end_time = df_end_time.reset_index()
df_end_time.index = df_end_time.timestamp

busy_days = df_end_time.groupby(["year", "month", "day"]).timestamp.count() > 100
busy_days = busy_days[busy_days==True].reset_index()[["year", "month", "day"]]
busy_days = pd.to_datetime(busy_days).astype(str)

#For every busiest days of the dataset (more than 200 movements)
#Get the time difference between two consecutives flights
dt_sec = {}
for i in tqdm(busy_days):
    delta_t = np.diff(df_end_time.loc[i].between_time('04:00','20:00').timestamp)
    helper = np.vectorize(lambda x: x.total_seconds())
    dt_sec[i] = helper(delta_t)
    
#%%
agg_delta_t = np.array([x for v in dt_sec.values() for x in v])
x = agg_delta_t

#Birnbaum-Saunders distribution, which is typically used to model failure times
dist_fl = st.fatiguelife
args_fl = dist_fl.fit(x)
print(args_fl)
print(st.kstest(x, dist_fl.cdf, args_fl))

#%%
plt.style.use("bmh")

seconds = np.linspace(x.min(), x.max(), 5000)
dist = dist_fl.pdf(seconds, *args_fl)
nbins = 100
fig, ax = plt.subplots(1, 1, figsize=(15,7))
ax.hist(x, nbins, alpha=0.5, label="Observed", density = True, color = "orange")
ax.plot(seconds, dist,
        '--g', lw=3, label='Birnbaum-Saunders')

#Quantiles
quant_5, quant_25, quant_50, quant_75, quant_95 = np.quantile(x, 0.05), np.quantile(x, 0.25), np.quantile(x, 0.5), np.quantile(x, 0.75), np.quantile(x, 0.95)
quants = [[quant_5, 0.6, 0.10], [quant_25, 0.8, 0.20], [quant_50, 1, 0.30],  [quant_75, 0.8, 0.46], [quant_95, 0.6, 0.56]]
for i in quants:
    ax.axvline(i[0], alpha = i[1], ymax = i[2], linestyle = ":")

ax.text(quant_5-30, 0.0003, "$5^{th}$", size = 20, alpha = 1)
ax.text(quant_25-50, 0.0006, "$25^{th}$", size = 20, alpha = 1)
ax.text(quant_50-80, 0.00085, "$50^{th}$", size = 20, alpha = 1)
ax.text(quant_75-50, 0.00132, "$75^{th}$", size = 20, alpha = 1)
ax.text(quant_95-50, 0.0016, "$95^{th}$", size = 20, alpha =1)

ax.set_xlabel("Time Separation (s)", size = 18)
ax.tick_params(axis='x', labelsize=15)
ax.set_xlim(-10, 3000)
ax.grid(False)

ax.set_yticks([])
ax.set_ylabel("")

ax.set_title("Time Separation Distribution between Consecutive Aircrafts", size = 22, pad = 10)
ax.tick_params(left = False, bottom =True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.legend(fontsize=17, frameon=False)
plt.show()