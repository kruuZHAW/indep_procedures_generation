# %%
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from os import walk

# %%
dataset_ldng = TrafficDataset.from_file(
    # "../deep_traffic_generation/data/training_datasets/landings_LFPO_06.pkl",
    "../deep_traffic_generation/data/training_datasets/landings_south_LFPO_25.pkl",
    features=["track", "groundspeed", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1, 1)),
    shape="image",
    info_params={"features": ["latitude", "longitude"], "index": -1},
)

dataset_to = TrafficDataset.from_file(
    # "../deep_traffic_generation/data/training_datasets/takeoffs_LFPO_07.pkl",
    "../deep_traffic_generation/data/training_datasets/takeoffs_south_proc_LFPO_24.pkl",
    features=["track", "groundspeed", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1, 1)),
    shape="image",
    info_params={"features": ["latitude", "longitude"], "index": -1},
)

#East
# path_to = "../deep_traffic_generation/lightning_logs/tcvae/version_0/"
# path_ldng = "../deep_traffic_generation/lightning_logs/tcvae/version_1/"

#West
path_to = "../deep_traffic_generation/lightning_logs/tcvae/version_4/"
path_ldng = "../deep_traffic_generation/lightning_logs/tcvae/version_5/"

t_to = SingleStageVAE(X=dataset_to, sim_type="generation")
t_to.load(path_to, dataset_to.parameters)
g_to = Generation(
    generation=t_to,
    features=t_to.VAE.hparams.features,
    scaler=dataset_to.scaler,
)

t_ldng = SingleStageVAE(X=dataset_ldng, sim_type="generation")
t_ldng.load(path_ldng, dataset_ldng.parameters)
g_ldng = Generation(
    generation=t_ldng,
    features=t_ldng.VAE.hparams.features,
    scaler=dataset_ldng.scaler,
)

# %%
pX_to = t_to.VAE.lsr.pseudo_inputs_NN(t_to.VAE.lsr.idle_input)
pX_to = pX_to.view((pX_to.shape[0], 4, 200))
ph_to = t_to.VAE.encoder(pX_to)
pmeans_to = t_to.VAE.lsr.z_loc(ph_to)
pscales_to = (t_to.VAE.lsr.z_log_var(ph_to) / 2).exp()

pX_ldng = t_ldng.VAE.lsr.pseudo_inputs_NN(t_ldng.VAE.lsr.idle_input)
pX_ldng = pX_ldng.view((pX_ldng.shape[0], 4, 200))
ph_ldng = t_ldng.VAE.encoder(pX_ldng)
pmeans_ldng = t_ldng.VAE.lsr.z_loc(ph_ldng)
pscales_ldng = (t_ldng.VAE.lsr.z_log_var(ph_ldng) / 2).exp()

# %%
#EAST
# j_to = 488
# j_ldng = 273
#WEST
j_to = 233
j_ldng = 436
n_gen = 100

dist_to = torch.distributions.Independent(
    torch.distributions.Normal(pmeans_to[j_to], pscales_to[j_to]), 1
)
gen_to = dist_to.sample(torch.Size([n_gen]))
decode_to = t_to.decode(
    torch.cat((pmeans_to[j_to].unsqueeze(0), gen_to), axis=0)
)
decode_to[:, 3] = 0
traf_gen_to = g_to.build_traffic(
    decode_to,
    # coordinates=dict(latitude=48.736157, longitude = 2.45031), #East
    coordinates=dict(latitude=48.71687, longitude = 2.308608), #West
    forward=True,
)

dist_ldng = torch.distributions.Independent(
    torch.distributions.Normal(pmeans_ldng[j_ldng], pscales_ldng[j_ldng]), 1
)
gen_ldng = dist_ldng.sample(torch.Size([n_gen]))
decode_ldng = t_ldng.decode(
    torch.cat((pmeans_ldng[j_ldng].unsqueeze(0), gen_ldng), axis=0)
)
decode_ldng[:, 3] = 0
traf_gen_ldng = g_ldng.build_traffic(
    decode_ldng,
    # coordinates=dict(latitude=48.704496, longitude = 2.273339), #East
    coordinates=dict(latitude=48.736236, longitude = 2.449810), #West
    forward=False,
)

# traf_gen_to.to_pickle("generated_to_east.pkl")
# traf_gen_ldng.to_pickle("generated_ldng_east.pkl")
traf_gen_to.to_pickle("generated_to_west.pkl")
traf_gen_ldng.to_pickle("generated_ldng_west.pkl")
# %%
