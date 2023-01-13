"""Script to smooth altitude in generated trajectories in order to compute vertical rate
Use: savgol filter
"""

from pathlib import Path
import click
import time
import os

from tqdm.autonotebook import tqdm 
from traffic.core import Traffic
from scipy.signal import savgol_filter

def processing_vertical_rate(f, w, d):
    if (len(f.data) <= 10):
        if len(f.data) > d:
            f.data.altitude = savgol_filter(f.data.altitude, len(f.data), d)
            f.data.groundspeed = savgol_filter(f.data.groundspeed, len(f.data), d)
    else:
        f.data.altitude = savgol_filter(f.data.altitude, w, d)
        f.data.groundspeed = savgol_filter(f.data.groundspeed, w, d)
    f = f.assign(vertical_rate=lambda df: df.altitude.diff().fillna(0) * 60 / 5)
    return f

@click.command()
@click.argument("traffic_path", type=click.Path(exists=True))
@click.argument("name", type=str)


def main(
    traffic_path:  Path,
    name:  str,
):

    start_time = time.time()

    click.echo("Loading Generated Traffics...")
    t = Traffic.from_file(traffic_path)
    click.echo("--- %s seconds ---" % (time.time() - start_time))
    
    click.echo("Processing...")
    t = t.iterate_lazy().pipe(processing_vertical_rate, 10, 3).eval(desc = "", max_workers = 100)
    click.echo("--- %s seconds ---" % (time.time() - start_time))
    
    click.echo("Saving results...")
    t.to_parquet(name)
    click.echo("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()