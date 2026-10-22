import os
import glob
from datetime import datetime
import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import cmcrameri as cmcr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from argparse import ArgumentParser

TIME_FMT = '%Y%m%d-%H:%M:%S'
SEC_PER_DAY = 86400.0
KG_PER_TG = 1e9

'''
computes total fluxes across multiple control surfaces,
and plots maps of horizontal fluxes,
all of this for of user-defined variables (aggregated)

NB: in the model output files each flux is a MEAN RATE,
    sum_t(C * v * A * dt) / elapsed_time  [tracer unit * m3 s-1],
    e.g. [mg C s-1] for all the carbon tracers.
    Fluxes are integrated over the cell face area, not per unit area.
    - A file with attributes Time_Start / Time___End holds the flux accumulated
      since the previous dump (the accumulator is reset at every dump), but it is divided by
      (Time___End - Time_Start). Time_Start is the last *regular* dump: after a
      restart-time backup dump it is NOT updated, so the file written after the backup
      covers only (backup time, Time___End) while being normalised over the whole window.
      Hence rate * (Time___End - Time_Start) is always the integral over the accumulation
      interval (End of previous file, End of this file). This holds for any output
      frequency (daily, weekly, monthly, with or without backup dumps).
    - the time mean over a period is the time-weighted mean of these accumulation
      intervals, clipped to the period.
    The cells of a control surface are located through the 'index' variable of each
    flux file (its order differs from Fluxes.nc).
    Results are in [kg y-1] per cell; the printed totals are in [Tg y-1].

Usage: 
source /g100/home/userexternal/ggalli00/PY_ENVS/bitsea_gg/sequence3.sh
python analyse_fluxes.py -y analyse_fluxes.yaml
'''

def load_parameters():
    parser = ArgumentParser()
    parser.add_argument(
        "--yamlfile", "-y",
        required=True,
        help="path to file with all parameters"
    )
    args = parser.parse_args()
    yamlfile = args.yamlfile
    # yamlfile = 'analyse_fluxes.yaml'
    with open(yamlfile) as f:
        Params = yaml.safe_load(f)
    return Params

def flux_components():
    '''
    index of flux components in OGSTM
    and on which grids they are defined
    'Type' is poor wording but is the same in model output files
    '''
    C = {
         'Zonal_advection':        {'Type': 0, 'grid': 'U'},
         'Meridional_advection':   {'Type': 1, 'grid': 'V'},
         'Vertical_advection':     {'Type': 2, 'grid': 'W'},
         'Gravitational_settling': {'Type': 3, 'grid': 'W'},
         'Horizontal_diffusion_x': {'Type': 4, 'grid': 'U'},
         'Horizontal_diffusion_y': {'Type': 5, 'grid': 'V'},
         'Vertical_diffusion':     {'Type': 6, 'grid': 'W'},
         'Bottom_burial':          {'Type': 7, 'grid': 'W'},
         }
    return C

def load_bounmask(Params):
    '''
    load bounmask and its shape
    '''
    with xr.open_dataset(Params['bounmask']) as ds:
        bmask_index = ds['index'].values[0]  # (jpk, jpj, jpi), NaN at land points
    jpk, jpj, jpi = bmask_index.shape
    return bmask_index, jpk, jpj, jpi

def load_surf_i(flx_params, Params):
    '''
    reads one control surface from its own *.nc file.
    Returns a dictionary with the bounmask indexes of the cells in the surface
    ('index') and equally shaped arrays U, V, W.
    These are multipliers (0 | 1 | -1) for sign conventions of actual model output.
    All arrays follow the order of the control surface file.
    The position of these cells in the model output is found for each flux file
    with surface_positions(), because the order of 'index' in the flux files
    (MPI gather order) differs from the one in Fluxes.nc.
    '''
    flxdir = Params['flxdir']
    with xr.open_dataset(f"{flxdir}{flx_params['ctrlsrf']}") as F:
        surf_idx = {'index': F['index'].values.astype(np.int64),
                    'U': F['U'].values.astype(np.int64),
                    'V': F['V'].values.astype(np.int64),
                    'W': F['W'].values.astype(np.int64)}
    if len(np.unique(surf_idx['index'])) != len(surf_idx['index']):
        raise ValueError(f"duplicated indexes in {flx_params['ctrlsrf']}")
    return surf_idx

def surface_positions(file_index, srf_index):
    '''
    positions in the model output (dimension n) of the cells of a control surface,
    in the order of srf_index. file_index is the 'index' variable of the flux file.
    '''
    sorter = np.argsort(file_index)
    loc = np.searchsorted(file_index, srf_index, sorter=sorter)
    loc = np.clip(loc, 0, len(file_index) - 1)
    pos = sorter[loc]
    missing = file_index[pos] != srf_index
    if missing.any():
        raise ValueError(f'{missing.sum()} control surface cells not found in the flux file')
    return pos

def parse_time(s):
    return datetime.strptime(s, TIME_FMT)

def list_flux_files(Params):
    '''
    Lists the model output flux files in chronological order, with the time interval
    over which each file accumulated its fluxes (see docstring on top):
    t_start, t_end : Time_Start, Time___End attributes (normalisation window)
    t_acc0, t_end  : accumulation interval (end of the previous file, end of this file)
    Works with any output frequency.
    '''
    indir = Params['indir'].replace('__TEST_NAME__', Params['test_name'])
    flist = glob.glob(indir + Params.get('fname', 'flux.*.nc'))
    if len(flist) == 0:
        raise FileNotFoundError(f'no flux files in {indir}')
    info = []
    for fname in flist:
        with xr.open_dataset(fname) as F:
            info.append({'fname': fname,
                         't_start': parse_time(F.attrs['Time_Start']),
                         't_end': parse_time(F.attrs['Time___End'])})
    info.sort(key=lambda x: x['t_end'])
    prev_end = info[0]['t_start']
    for x in info:
        if x['t_start'] > prev_end:
            print(f"WARNING: gap before {os.path.basename(x['fname'])}: "
                  f"{prev_end} -> {x['t_start']} (missing files?)")
        x['t_acc0'] = max(prev_end, x['t_start'])
        prev_end = x['t_end']
    return info

def get_period(Params, info):
    '''
    averaging period, from period_start and period_end in the .yaml file
    (null: first Time_Start, last Time___End of the available files)
    '''
    t0 = Params.get('period_start')
    t1 = Params.get('period_end')
    t0 = parse_time(t0) if t0 else min(x['t_start'] for x in info)
    t1 = parse_time(t1) if t1 else max(x['t_end'] for x in info)
    return t0, t1

def load_one_flux_file(fname, flx_params, surf_idx, Params):
    '''
    From one model output file,
    loads one flux at a control surface, 
    as defined in the .yaml file.
    Sums up all flux variables and components
    at each grid cell in the control surface.
    Returns a np.array [mg s-1] (rate as stored in the file, i.e. normalised over
    Time___End - Time_Start), with the order of the control surface file
    '''
    C = flux_components()
    flux_types = flx_params['flux_types']
    bfmvars = flx_params['bfmvars']
    with xr.open_dataset(fname) as F:
        pos = surface_positions(F['index'].values.astype(np.int64), surf_idx['index'])
        Flux = np.zeros(len(pos))
        # loop on flux types and model variables
        for ff in flux_types:
            iflx = C[ff]['Type'] # which flux component, 0:8
            gflx = C[ff]['grid'] # on which grid, U|V|W
            sign = surf_idx[gflx] # flux sign
            for vv in bfmvars:
                # extract flux component and control surface
                Flux += F[vv].isel(Type=iflx).values[pos] * sign
    return Flux

def load_all_flux_files(flx_params, surf_idx, Params, info):
    '''
    Loop on the model output files that overlap the averaging period,
    loads one flux at a control surface,
    as defined in its own .yaml file.
    Takes the time-weighted mean over the period, 
    each file being a mean over its accumulation interval (see docstring on top),
    and converts it to [kg y-1].
    Returns the flux in each cell of the control surface and
    the number of days of the period covered by the files.
    '''
    t0, t1 = get_period(Params, info)
    # initialise
    Flux = 0.0
    covered = 0.0 # [s]
    for x in info:
        a = max(x['t_acc0'], t0)
        b = min(x['t_end'], t1)
        overlap = (b - a).total_seconds()
        if overlap <= 0:
            continue
        window = (x['t_end'] - x['t_start']).total_seconds()
        acc_len = (x['t_end'] - x['t_acc0']).total_seconds()
        # file rate * window = integral over the accumulation interval
        scale = window / acc_len * overlap
        Flux = Flux + load_one_flux_file(x['fname'], flx_params, surf_idx, Params) * scale
        covered += overlap
    if covered == 0:
        raise ValueError(f'no flux files in the period {t0} - {t1}')
    Flux = Flux / covered  # time mean [mg s-1]
    # convert to [kg / y]
    sec_per_year = Params.get('days_per_year', 365.0) * SEC_PER_DAY
    Flux = Flux * sec_per_year  #[mg s-1 -> mg y-1]
    Flux = Flux / 1e06          #[mg y-1 -> kg y-1]
    return Flux, covered / SEC_PER_DAY

### PLOTTING

def unravel_flux(surf_idx, Flux, Params):
    indexes = surf_idx['index']
    bmask_index, jpk, jpj, jpi = load_bounmask(Params)
    Flux_map = np.full((jpj, jpi), np.nan)
    for idx, flx in zip(indexes, Flux):
        _, curr_j, curr_i = np.where(bmask_index == idx)
        Flux_map[curr_j, curr_i] = flx
    return Flux_map

def plot_flux_map(surf_idx, Flux, Params):
    Flux_map = unravel_flux(surf_idx, Flux, Params)
    return

if __name__ == '__main__':
    Params = load_parameters()
    info = list_flux_files(Params)
    t0, t1 = get_period(Params, info)
    print(f'averaging period: {t0} - {t1} ({(t1 - t0).total_seconds() / SEC_PER_DAY:.2f} days)')
    flx_list = Params['FLUXES'].keys()
    for ffi in flx_list:
        print(f'computing: {ffi}')
        flx_params = Params['FLUXES'][ffi]
        surf_idx = load_surf_i(flx_params, Params)
        Flux, days_covered = load_all_flux_files(flx_params, surf_idx, Params, info)
        if abs(days_covered - (t1 - t0).total_seconds() / SEC_PER_DAY) > 1e-6:
            print(f'WARNING: files cover only {days_covered:.2f} days of the period')
        #
        totF = np.sum(Flux) / KG_PER_TG
        print(f"{flx_params['name']} {flx_params['ctrlsrf']} \n {flx_params['flux_types']} \n total: {totF:.4f} Tg y-1")
        print('...')


