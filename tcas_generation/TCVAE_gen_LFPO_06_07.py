from deep_traffic_generation.tcvae import TCVAE
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDataset
from traffic.core import Traffic

from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from os import walk
import click
from pathlib import Path
import time

from torch.distributions import (
    Distribution, Independent, MixtureSameFamily, MultivariateNormal, Normal
)
from torch.distributions.categorical import Categorical

#loops
def simple(flight):
    return flight.assign(simple=lambda x: flight.shape.is_simple)

def load_TCVAE(dataset_path : Path, version:  str)-> tuple:

    dataset = TrafficDataset.from_file(
        dataset_path,
        features=["track", "groundspeed", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1,1)),
        shape="image",
        info_params={"features": ["latitude", "longitude"], "index": -1},
    )

    tcvae_path = "../deep_traffic_generation/lightning_logs/tcvae/"+ version + "/"

    t = SingleStageVAE(X = dataset, sim_type = "generation")
    t.load(tcvae_path, dataset.parameters)
    g = Generation(generation=t, features = t.VAE.hparams.features, scaler=dataset.scaler) 

    return t, g

def pseudo_inputs(t: SingleStageVAE, g: Generation, lat:float, lon:float, forward:bool) -> tuple:
    #Vampprior
    pseudo_X = t.VAE.lsr.pseudo_inputs_NN(t.VAE.lsr.idle_input) 
    pseudo_X = pseudo_X.view((pseudo_X.shape[0], 4, 200))

    pseudo_h = t.VAE.encoder(pseudo_X)
    pseudo_means = t.VAE.lsr.z_loc(pseudo_h)
    pseudo_scales = (t.VAE.lsr.z_log_var(pseudo_h) / 2).exp()

    #Reconstructed pseudo-inputs
    out = t.decode(pseudo_means)
    #Neural net don't predict exaclty timedelta = 0 for the first observation
    out[:,3] = 0
    out_traf = g.build_traffic(out,coordinates = dict(latitude = lat, longitude = lon), forward=forward)

    return out_traf, pseudo_means, pseudo_scales

#Based on the analysis of the pseudo-inputs in a notebook
def selecting_PI_06(pi:Traffic)-> Traffic:

    first_track = pi.data.groupby("flight_id")["track"].first()
    id_from_south = first_track[first_track > 270].index
    selected_PI = pi[id_from_south]

    selected_PI = selected_PI.iterate_lazy().pipe(simple).eval(desc ="")
    selected_PI = selected_PI.query("simple")

    selected_PI = selected_PI.query(
        "flight_id not in ['TRAJ_506', 'TRAJ_129', 'TRAJ_73', 'TRAJ_123', 'TRAJ_99', 'TRAJ_94', 'TRAJ_90']"
    )

    id_PI = [int(i.split("_",1)[1]) for i in selected_PI.flight_ids]
    return id_PI

def selecting_PI_07(pi:Traffic)-> Traffic:

    last_track = pi.data.groupby("flight_id")["track"].last()
    id_to_south = last_track[(last_track > 130) & (last_track < 210)].index
    selected_PI = pi[id_to_south]

    selected_PI = selected_PI.iterate_lazy().pipe(simple).eval(desc ="")
    selected_PI = selected_PI.query("simple")

    selected_PI = selected_PI.query(
    "flight_id not in ['TRAJ_286', 'TRAJ_269', 'TRAJ_590']"
    )

    id_PI = [int(i.split("_",1)[1]) for i in selected_PI.flight_ids]
    return id_PI

def generate_traffic(selected_pseudo_means:torch.Tensor, selected_pseudo_scales:torch.Tensor, t: SingleStageVAE, g: Generation, lat:float, lon:float, forward:bool, n:int)->Traffic:

    dist_gen_landing_tcas = MixtureSameFamily(
        Categorical(torch.ones((len(selected_pseudo_means),))),
            Independent(
                Normal(
                    selected_pseudo_means,
                    selected_pseudo_scales,
                ),
                1,
            ),
    )

    gen_latent_tcas = dist_gen_landing_tcas.sample(torch.Size([n]))
    decode_tcas = t.decode(gen_latent_tcas)
    decode_tcas[:, 3] = 0

    traf_gen_tcas = g.build_traffic(
    decode_tcas,
    coordinates = dict(latitude = lat, longitude = lon),
    forward=forward,
    )

    return traf_gen_tcas


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("version", type=str)
@click.argument("lat", type=float)
@click.argument("lon", type=float)
@click.option("-n", "--n-gen", default=100000, help="Number of generated trajectories")

def main(
    dataset_path:  Path,
    version:  str,
    lat:  float,
    lon: float,
    n_gen:int,
):

    start_time = time.time()
    click.echo("Loading TCVAE...")
    t, g = load_TCVAE(dataset_path, version)
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    click.echo("Selecting suitable pseudo-inputs...")
    if dataset_path.split("/")[-1] == "takeoffs_LFPO_07.pkl": 
        out_traf, pseudo_means, pseudo_scales = pseudo_inputs(t, g, lat, lon, forward = True)
        id_PI = selecting_PI_07(out_traf)
        click.echo("--- %s seconds ---" % (time.time() - start_time))
        click.echo("Generating traffic...")
        gen_traf = generate_traffic(pseudo_means[id_PI], pseudo_scales[id_PI], t, g, lat, lon, True, n_gen)

    elif dataset_path.split("/")[-1] == "landings_LFPO_06.pkl":
        out_traf, pseudo_means, pseudo_scales = pseudo_inputs(t, g, lat, lon, forward = False)
        id_PI = selecting_PI_06(out_traf)
        click.echo("--- %s seconds ---" % (time.time() - start_time))
        click.echo("Generating traffic...")
        gen_traf = generate_traffic(pseudo_means[id_PI], pseudo_scales[id_PI], t, g, lat, lon, False, n_gen)
        
    else:
        raise ValueError("Selection for those pseudo-inputs hasn't been implemented")
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    gen_traf.to_pickle("../deep_traffic_generation/data/generated_datasets/gen_"+ dataset_path.split("/")[-1])
    click.echo("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()