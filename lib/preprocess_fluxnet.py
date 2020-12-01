import math
import os.path
import random
import time
from glob import glob

import dask
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump, load
from tqdm import tqdm
from numba import jit
from numpy.lib.stride_tricks import as_strided
from joblib import Parallel, delayed

from dask.distributed import Client
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

read_vars = ['TA_F', 'P', 'SW_IN_F', 'LW_IN_F', 'PA_F', 'RH',
             'WS', 'LE_F_MDS', 'H_F_MDS', 'LE_CORR', 'H_CORR', 'NETRAD', 'G_F_MDS']
target_vars = read_vars
train_vars = read_vars
era_vars = ['TA_ERA', 'P_ERA', 'SW_IN_ERA', 'LW_IN_ERA', 'PA_ERA', 'VPD_ERA',
            'WS_ERA']


MIN_LENGTH = 3 * 17520
MAX_GAP_LENGTH = 72

# these missing in the metadata and were looked up using google earth
elevs = {
    "AR-Vir": 105.0,
    "AU-Wac": 753.0,
    "AR-SLu": 508.0,
    "AU-Rig": 151.0,
    "AU-Stp": 228.0,
    "CN-Du2": 1321.0,
    "JP-SMF": 213.0,
    "AU-How": 39.0,
    "AU-DaP": 71.0,
    "CN-Sw2": 1445.0,
    "AU-Dry": 176.0,
    "AU-Emr": 175.0,
    "CN-Din": 292.0,
    "AU-DaS": 74.0,
    "CN-Cng": 143.0,
    "AU-Whr": 151.0,
    "AU-Fog": 4.0,
    "AU-RDF": 189.0,
    "RU-Sam": 11.0,
    "AU-Cum": 39.0,
    "CN-Qia": 112.0,
    "CN-Du3": 1313.0,
    "CN-Ha2": 3198.0,
    "CN-Cha": 767.0,
    "AU-Gin": 51.0,
    "AU-Ade": 76.0,
    "CN-HaM": 4004.0,
    "AU-GWW": 448.0,
    "AU-Ync": 126.0,
    "JP-MBF": 572.0,
    "MY-PSO": 147.0,
    "AU-TTE": 552.0,
    "AU-ASM": 606.0,
    "CN-Dan": 4313.0,
    "AU-Cpr": 63.0,
    "AU-Lox": 45.0,
    "AU-Rob": 710.0,
}


sites_to_skip = [
    "CA-Man",  # missing RH
    "DE-RuR",  # missing RH
    "CA-Man",  # missing RH
    "DE-RuR",  # missing RH
    "DE-RuS",  # missing RH
    "MY-PSO",  # missing RH
#     "CN-Cha",  # found nans in df
#     "CN-Dan",  # found nans in df
#     "CN-Din",  # found nans in df
#     "CN-Qia",  # found nans in df
#     "DK-ZaH",  # found nans in df
#     "FI-Lom",  # found nans in df
#     "IT-Isp",  # found nans in df
#     "IT-SR2",  # found nans in df
#     "US-Me5",  # found nans in df
#     "US-PFa",  # found nans in df
]


def get_fluxnet(cat, from_cache=True):
    """load the fluxnet dataset"""
    if not from_cache:
        # use dask to speed things up
        x_data_computed, y_data_computed, meta = load_fluxnet(cat)
    else:
        x_data_computed = load("./etl_data/x_data_computed.joblib")
        y_data_computed = load("./etl_data/y_data_computed.joblib")
        meta = load("./etl_data/meta.joblib")

    return x_data_computed, y_data_computed, meta


@dask.delayed
def load_fluxnet_site(entry, name):
    try:
        df = entry.read().set_index("TIMESTAMP_END")
        return df
    except:
        print(f'Failed to load {name}')
        return None


def add_meta(df, meta):
    df["lat"] = meta["lat"]
    df["lon"] = meta["lon"]
    df["elev"] = meta["elev"]
    return df


def get_meta(all_site_meta):
    all_sites = all_site_meta.index.get_level_values(0).unique()
    meta = {
        key: extract_site_meta(all_site_meta, key)
        for key in all_sites
        if key not in sites_to_skip
    }
    return meta


def load_fluxnet(cat, all_site_meta):
    """return lists of x and y data"""
    meta = get_meta(all_site_meta)
    meta_df = pd.DataFrame.from_dict(meta, orient="index")

    site_data = {}
    for site, site_meta in meta.items():
        site_data[site] = load_fluxnet_site(cat["subdaily"](site=site), site)

    site_data = dask.compute(site_data)[0]

    out = {}
    var_names = train_vars
    for name, df in site_data.items():
        if df is not None:
            out[name] = add_meta(df.loc[:, var_names], meta[name])

    return pd.concat(out.values(), keys=out.keys())


def load_era(cat, all_site_meta):
    """return lists of x and y data"""
    meta = get_meta(all_site_meta)
    meta_df = pd.DataFrame.from_dict(meta, orient="index")

    site_data = {}
    for site, site_meta in meta.items():
        site_data[site] = load_fluxnet_site(cat["subdaily"](site=site), site)

    site_data = dask.compute(site_data)[0]

    out = {}
    var_names = era_vars
    for name, df in site_data.items():
        if df is not None:
            out[name] = add_meta(df.loc[:, var_names], meta[name])

    return pd.concat(out.values(), keys=out.keys())


def first_entry(entry):
    try:
        return entry.astype(float).values[0]
    except:
        return float(entry)


def extract_site_meta(meta, site):
    out = {}
    out["lat"] = first_entry(meta[site]["LOCATION_LAT"])
    out["lon"] = first_entry(meta[site]["LOCATION_LONG"])

    try:
        out["elev"] = first_entry(meta[site]["LOCATION_ELEV"])
    except:
        try:
            out["elev"] = elevs[site]
        except KeyError:
            print(f"failed to get elevation for {site}")
    return out


def simple_get_longest_sequence(df, site_list, min_length=MIN_LENGTH, max_gap=MAX_GAP_LENGTH):
    all_lens = []
    out = {}
    for site in site_list:
        test_df = df.loc[site]
        sum_df = test_df.sum(axis=1, skipna=False)
        idx = sum_df.groupby((0 * sum_df).fillna(1/(max_gap+1)).cumsum().apply(np.floor)).transform('count')
        mymax = test_df[idx==idx.max()]
        if len(mymax) > min_length and idx.max() != idx.min():
            out[site] = mymax
    return pd.concat(out.values(), keys=out.keys())


def subsequences(arr, m):
    n = arr.size - m + 1
    s = arr.itemsize
    return as_strided(arr, shape=(m,n), strides=(s,s)).T


@jit(nopython=True)
def frac_nan_all_subsequences(arr, min_length=17520, granularity=48):
    max_len = len(arr)
    len_list = []
    frac_list = []
    start_list = []
    for m in range(min_length, max_len+1, granularity):
        n = arr.size - m + 1
        s = arr.itemsize
        subs = as_strided(arr, shape=(m, n), strides=(s,s)).T
        frac = np.sum(subs, axis=1) / m
        frac_list.append(frac)
        len_list.append(m * np.ones_like(frac))
        start_list.append(np.arange(len(frac)))
    return frac_list, len_list, start_list


def return_longest_subsequence(df, min_length=MIN_LENGTH, good_frac=0.9):
    sum_df = (~df.sum(axis=1, skipna=False).isna()).astype(int)
    fl, ll, sl = frac_nan_all_subsequences(sum_df.values,
                                           min_length=min_length,
                                           granularity=30*48)
    try:
        fl = np.hstack(fl)
        ll = np.hstack(ll)
        sl = np.hstack(sl)
        mask = fl > good_frac
        if np.sum(mask) > 0:
            frac_good = fl[mask]
            len_good = ll[mask]
            start_good = sl[mask]
            start = int(start_good[-1])
            end = int(start_good[-1] + len_good[-1])
            return df[start:end]
    except:
        return None
    return None


def get_longest_sequence(df, site_list, n_workers=1, min_length=MIN_LENGTH, good_frac=0.9):
    out = Parallel(n_jobs=n_workers)(
            delayed(return_longest_subsequence)(df.loc[s], min_length, good_frac)
            for s in site_list)
    ol = []
    sl = []
    for o, s in zip(out, site_list):
        if o is not None:
            sl.append(s)
            ol.append(o)
    return pd.concat(ol, keys=sl)


def plot_sites_per_gap_length(df, all_sites):
    gap_lens = [1, 48, 96, 192, 512, 1752, 2 * 1752, 3 * 1752]
    all_dfs = [get_longest_sequence(df, all_sites, min_length=MIN_LENGTH, max_gap=gl) for gl in gap_lens]
    n_sites = [len(np.unique(df.index.get_level_values(0))) for df in all_dfs]

    plt.bar(np.arange(len(gap_lens)), n_sites)
    plt.xticks(np.arange(len(gap_lens)), np.around(100*np.array(gap_lens)/MIN_LENGTH, 1), rotation=45)
    plt.xlabel('Percentage of missing data allowed')
    plt.ylabel('Number of sites')
    plt.axhline(25 , linewidth=1, color='grey')
    plt.axhline(50 , linewidth=1, color='grey')
    plt.axhline(75 , linewidth=1, color='grey')
    plt.axhline(100, linewidth=1, color='grey')
    plt.title(f'# Sites available if requiring {int(MIN_LENGTH/17520)} years of data')


def spechum(rh, T, p):
    """Calculate specific humidity"""
    T0 = 273.16
    return rh * np.exp((17.67*(T-T0))/(T-29.65)) / (0.263*p)


def svp(temp: np.array, a: float=0.61078, b: float=17.269, c: float=237.3):
    svp = a * np.exp((b * temp) / (c + temp))
    inds = np.nonzero(temp < 0.)[0]
    svp[inds] *= 1.0 + .00972 * temp[inds] + .000042 * np.power(temp[inds], 2)
    return svp * 10.  # convert to hpa


def relhum(t, vpd):
    return 100 * (1 - vpd / svp(t))


def merge_fluxnet_era(fluxnet_df, era_df):
    fluxnet_dates = fluxnet_df.index
    era_sub = era_df.loc[fluxnet_dates]

    filt = fluxnet_df['TA_F'].isna()
    fluxnet_df['TA_F'][filt] = era_sub['TA_ERA'][filt]

    filt = fluxnet_df['P'].isna()
    fluxnet_df['P'][filt] = era_sub['P_ERA'][filt]

    filt = fluxnet_df['SW_IN_F'].isna()
    fluxnet_df['SW_IN_F'][filt] = era_sub['SW_IN_ERA'][filt]

    filt = fluxnet_df['LW_IN_F'].isna()
    fluxnet_df['LW_IN_F'][filt] = era_sub['LW_IN_ERA'][filt]

    filt = fluxnet_df['PA_F'].isna()
    fluxnet_df['PA_F'][filt] = era_sub['PA_ERA'][filt]

    filt = fluxnet_df['WS'].isna()
    fluxnet_df['WS'][filt] = era_sub['WS_ERA'][filt]

    filt = fluxnet_df['RH'].isna()
    fluxnet_df['RH'][filt] = relhum(fluxnet_df['TA_F'][filt], era_sub['VPD_ERA'][filt])

    fluxnet_df['spechum'] = spechum(fluxnet_df['RH'], fluxnet_df['TA_F']+273.16, fluxnet_df['PA_F']*1000)
    return fluxnet_df


def to_summa_ds(df):
    lats = [df['lat'].values[0]]
    lons = [df['lon'].values[0]]
    elev = [df['elev'].values[0]]
    bounds = df.index
    bounds.name = 'time'

    shape = (len(bounds), 1, )
    dims = ('time', 'hru', )
    coords = {'time': bounds}
    met_data = xr.Dataset(coords=coords)

    attrs = {
        'airpres':    {'units': 'Pa', 'long_name': 'Air pressure'},
        'airtemp':    {'units': 'K', 'long_name': 'Air temperature'},
        'spechum':    {'units': 'g g-1', 'long_name': 'Specific humidity'},
        'windspd':    {'units': 'm s-1', 'long_name': 'Wind speed'},
        'SWRadAtm':   {'units': 'W m-2', 'long_name': 'Downward shortwave radiation'},
        'LWRadAtm':   {'units': 'W m-2', 'long_name': 'Downward longwave radiation'},
        'pptrate':    {'units': 'kg m-2 s-1', 'long_name': 'Precipitation rate'},
        'Qle':        {'units': 'W m-2', 'long_name': 'Latent heat flux'},
        'Qh':         {'units': 'W m-2', 'long_name': 'Sensible heat flux'} ,
        'Qle_cor':    {'units': 'W m-2', 'long_name': 'Latent heat flux (EBC)'},
        'Qh_cor':     {'units': 'W m-2', 'long_name': 'Sensible heat flux (EBC)'} ,
        'gap_filled': {'units': '-', 'long_name': 'Flag for gap filled data'},
        'NetRad':     {'W m-2': '-', 'long_name': 'Net radiation'},
        'GroundHeatFlux': {'W m-2': '-', 'long_name': 'Ground heat flux'},
    }

    summa_vars = ['airpres', 'airtemp', 'spechum', 'windspd', 'SWRadAtm',
                  'LWRadAtm', 'pptrate', 'Qle', 'Qh', 'Qle_cor', 'Qh_cor',
                  'gap_filled', 'NetRad', 'GroundHeatFlux']
    for varname in summa_vars:
        met_data[varname] = xr.DataArray(data=np.full(shape, np.nan),
                                         coords=coords, dims=dims,
                                         name=varname, attrs=attrs[varname])

    met_data['airpres'   ].loc[{'hru': 0}] = df['PA_F'] * 1000
    met_data['airtemp'   ].loc[{'hru': 0}] = df['TA_F'] + 273.16
    met_data['windspd'   ].loc[{'hru': 0}] = df['WS']
    met_data['SWRadAtm'  ].loc[{'hru': 0}] = df['SW_IN_F']
    met_data['LWRadAtm'  ].loc[{'hru': 0}] = df['LW_IN_F']
    met_data['pptrate'   ].loc[{'hru': 0}] = df['P'] / 1800
    met_data['spechum'   ].loc[{'hru': 0}] = (
        spechum(df['RH'], df['TA_F']+273.16, df['PA_F']*1000))
    met_data['Qle'       ].loc[{'hru': 0}] = df['LE_F_MDS']
    met_data['Qh'        ].loc[{'hru': 0}] = df['H_F_MDS']
    met_data['Qle_cor'   ].loc[{'hru': 0}] = df['LE_CORR']
    met_data['Qh_cor'    ].loc[{'hru': 0}] = df['H_CORR']
    met_data['gap_filled'].loc[{'hru': 0}] = df['gap_filled']
    return met_data


def gen_fit_slices(starts, ends):
    first_slice = (slice(starts[1], ends[-1]), slice(None))
    mid_slices = [(slice(starts[0], ends[i-1]), slice(starts[i+1], ends[-1]))
                  for i in np.arange(1, len(starts)-1)]
    last_slice = (slice(None), slice(starts[0], ends[-2]))
    return [first_slice] + mid_slices + [last_slice]


def gen_transform_slices(starts, ends):
    return [slice(s, e) for s, e in zip(starts, ends)]


def fill_chunk(fit_df, transform_df):
    estimator = RandomForestRegressor(n_estimators=10, n_jobs=8)
    imp = IterativeImputer(estimator=estimator, max_iter=5, random_state=0)
    imp.fit(fit_df)
    transformed = imp.transform(transform_df)
    imputed_df = pd.DataFrame(data=transformed, index=transform_df.index,
                              columns=transform_df.columns)
    return imputed_df


def gap_fill(df):
    N_FIT_CHUNKS = 2
    CHUNK_LEN = len(df) // N_FIT_CHUNKS
    CHUNK_STARTS = list(CHUNK_LEN * np.arange(N_FIT_CHUNKS))
    CHUNK_ENDS = CHUNK_STARTS[1:] + [-1]

    transform_slices = gen_transform_slices(CHUNK_STARTS, CHUNK_ENDS)
    fit_slices = gen_fit_slices(CHUNK_STARTS, CHUNK_ENDS)

    imputed_chunks = []
    for i in range(N_FIT_CHUNKS):
        tfs1, tfs2 = fit_slices[i]
        fit_df = pd.concat([df[tfs1], df[tfs2]])
        transform_df = df[transform_slices[i]]
        imputed_chunks.append(fill_chunk(fit_df, transform_df))

    imputed = pd.concat(imputed_chunks)
    imputed['gap_filled'] = df.isna().sum(axis=1) > 0
    return imputed


def populate_metadata(ds, attrs):
    for k, v in attrs.items():
        ds.attrs[k] = v
    return ds
