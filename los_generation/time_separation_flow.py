""" Parameterise flow of landings and trajectories with a time separation sampled within a Bernbaum-Saunders distribution
    
    Arguments:
        gen_ldng_path: path to generated landings
        gen_to_path: path to generated take-offs
        east: True if EAST configuration at LFPO. False if WEST configuration
        ref_date: reference date to start both flows of trajectories
        --bbox/--no-bbox: if the trajectories have to be cropped within the defined bounding box

    Returns:
        traffic object: concatenation of take-off and landing flows
"""


from pathlib import Path
import click
import time
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from shapely.geometry import Polygon
from tqdm.autonotebook import tqdm 
from traffic.core import Traffic


def same_stop(f, t_ref):
    f.data.timestamp = f.data.timestamp + pd.to_timedelta(t_ref - f.stop, unit ="s")
    return f

def create_delta_t(t: Traffic, dist: st.rv_continuous, ref_date: pd.Timestamp, landing: bool, bbox: bool, east: bool) -> Traffic:
    
    #landings end at the same time
    if landing:
        t = t.iterate_lazy().pipe(same_stop, ref_date).eval(desc = "stop time normalization", max_workers = 100)
    
    #takeoffs start at the same time
    else:
        t.data = t.data.assign(
            timestamp=pd.to_timedelta(t.data.timedelta, unit="s") + ref_date
        )
        
    t.data.sort_index(inplace = True)
    
    gen_dt = np.concatenate(([0],dist.rvs(len(t)-1)))
    cum_gen_dt = np.round(np.cumsum(gen_dt))
    #dupplicate dt times the number of observations for each flights
    cum_gen_dt_aug = []
    for td in cum_gen_dt:
        cum_gen_dt_aug.extend([td]*200) #Each generated trajectory has exactly 200 points
    
    t.data.timestamp = t.data.timestamp + pd.to_timedelta(cum_gen_dt_aug, unit = "s")
    
    if landing:
        t = t.assign(flight_id = lambda df: "LDNG_" + df.flight_id)
    else:
        t = t.assign(flight_id = lambda df: "TO_" + df.flight_id)
    t = t.resample("5s").eval(desc = "resampling 5s", max_workers = 100)
    
    if bbox:
        # BBOX-EAST
        if east : 
            lat = [48.6102599, 48.267231, 48.267231, 48.6102599]
            lon = [2.163190, 2.163190, 2.82, 2.82]
        # BBOX-WEST
        else:
            lat = [48.70, 48.30, 48.30, 48.70]
            lon = [2.0,2.0,2.7,2.7]
        poly = Polygon(zip(lon, lat))
        t = t.inside_bbox((poly))
        
    return t
        

@click.command()
@click.argument("gen_ldng_path", type=click.Path(exists=True))
@click.argument("gen_to_path", type=click.Path(exists=True))
@click.argument("ref_date", type=str)
@click.option('--east/--west', default=True)
@click.option('--bbox/--no-bbox', default=True)

def main(
    gen_ldng_path:  Path,
    gen_to_path:  Path,
    ref_date:  str,
    east: bool,
    bbox: bool,
):

    start_time = time.time()

    ref_date = pd.Timestamp(ref_date, tz = 'UTC')
    if east : 
        dist = st.fatiguelife(c = 1.146240535025501, loc = 48.7344285250812, scale = 275.39162289183787) #Same distribution for takeoffs and landings
    else :
        dist = st.fatiguelife(c = 1.1635939736232785, loc = 52.634030248193504, scale = 251.13235704363382) #Same distribution for takeoffs and landings
        

    click.echo("Loading Generated Traffics...")
    gen_to = Traffic.from_file(gen_to_path)
    gen_ldng = Traffic.from_file(gen_ldng_path)
    click.echo("--- %s seconds ---" % (time.time() - start_time))
    
    click.echo("Calulating time difference bewtween consectuive flights for landings...")
    gen_ldng_dt = create_delta_t(gen_ldng, dist, ref_date, True, bbox=bbox, east=east)
    click.echo("--- %s seconds ---" % (time.time() - start_time))
    
    click.echo("Calulating time difference bewtween consectuive flights for takeoffs...")
    gen_to_dt = create_delta_t(gen_to, dist, ref_date, False, bbox=bbox, east=east)
    click.echo("--- %s seconds ---" % (time.time() - start_time))
    
    click.echo("Saving results...")
    num = len(glob.glob1(os.getcwd(),"*.parquet"))
    gen_traf = gen_to_dt + gen_ldng_dt
    gen_traf = gen_traf.assign(flight_id = lambda df: df.flight_id + "_MC_"+ str(num)) #Unique flight_ids over different MC traffics
    gen_traf.to_parquet("traffic_MC_" + str(num) + ".parquet")
    click.echo("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()