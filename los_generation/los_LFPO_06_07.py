    """Calculate LoS of separation for pair centered Monte Carlo simulations
    
    ***DEPRECIATED***

    Returns:
        pandas DataFrame: DataFrame with flight_ids of the pair, the time difference between landing and takeoff, and if a los occurred 
    """


from pathlib import Path
import click
import time
from multiprocessing import Pool
import os

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from traffic.core import Traffic
import pandas as pd
import numpy as np
import json

def delta_t_estimation(delta_t_path: Path) -> KernelDensity: 
    delta_t = pd.read_pickle(delta_t_path)

    bandwidths = np.logspace(-1, 3, 50)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=10)
    grid.fit(delta_t.delta_t.values[:, None])
    bw = grid.best_estimator_.bandwidth
    bw = 19.3
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(delta_t.delta_t.values[:,None])

    return kde

def simulate_los(iter):
    t, l, d = iter

    l.data = l.data.assign(
        timestamp = l.data.timestamp + pd.to_timedelta(d[0], unit="s") 
        )

    l = l.resample("1s")
    t = t.resample("1s")

    dist = t.distance(l)
    
    if dist is not None : 
        if dist.query("(lateral < 5) & (vertical < 1000)").shape[0] > 0:
            return {"id_to":t.flight_id, "id_ldng":l.flight_id, "delta_t":d[0], "los": True}
        else:
            return {"id_to":t.flight_id, "id_ldng":l.flight_id, "delta_t":d[0], "los": False}

    else :
        return {"id_to":t.flight_id, "id_ldng":l.flight_id, "delta_t":d[0], "los": False}




@click.command()
@click.argument("landing_path", type=click.Path(exists=True))
@click.argument("takeoff_path", type=click.Path(exists=True))
@click.argument("delta_t_path", type=click.Path(exists=True))


def main(
    landing_path:  Path,
    takeoff_path:  Path,
    delta_t_path:  Path,
):

    start_time = time.time()
    click.echo("Processing delta_t distribution estimation...")
    kde = delta_t_estimation(delta_t_path)
    click.echo("--- %s seconds ---" % (time.time() - start_time))
    
    click.echo("Charging data...")
    gen_to = Traffic.from_file(takeoff_path)
    gen_ldng = Traffic.from_file(landing_path)
    delta_t = kde.sample(len(gen_to))
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    click.echo("Checking for unified starting times between landings and takeoffs...")
    assert len(gen_to) == len(gen_ldng)
    start_to = gen_to.data.groupby("flight_id").timestamp.first().unique()
    start_ldng = gen_ldng.data.groupby("flight_id").timestamp.first().unique()
    assert len(start_to) == 1
    assert len(start_ldng) == 1
    assert start_to == start_ldng
    click.echo("Number of simulations: %s" % (len(gen_to)))
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    # Fastest way to do it: sans remise
    #The number of simulation is directly linked to the size of gen_to and gen_ldng
    click.echo("Processing Loss of Separation...")
    with Pool(processes=os.cpu_count()) as p: 
        los = p.map(simulate_los, zip(gen_to, gen_ldng, delta_t))
        p.close()
        p.join()    
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    click.echo("Saving results...")
    with open('los_LFPO_06_07.json', 'w') as fout:
        json.dump(los , fout)
    click.echo("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()