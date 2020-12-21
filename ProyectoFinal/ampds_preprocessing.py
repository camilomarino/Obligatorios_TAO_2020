#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import nilmtk


MINUTES_DAY = 60*24

FULL_NAME = {
             'WHE': 'Whole-House Meter',
             'B1E': 'North Bedroom',
             'B2E': 'Master and South Bedroom',
             'BME': 'Basement Plugs and Lights',
             'CDE': 'Clothes Dryer',
             'CWE': 'Clothes Washer',
             'DNE': 'Dining Room Plugs',
             'DWE': 'Dishwasher',
             'EBE': 'Electronics Workbench',
             'EQE': 'Security/Network Equipment',
             'FGE': 'Kitchen Fridge',
             'FRE': 'Forced Air Furnace: Fan and Thermostat',
             'GRE': 'Garage',
             'HPE': 'Heat Pump',
             'HTE': 'Instant Hot Water Unit',
             'OFE': 'Home Office',
             'OUE': 'Outside Plug',
             'RSE': 'Rental Suite',
             'TVE': 'Entertainment: TV, PVR, AMP',
             'UTE': 'Utility Room Plug',
             'WOE': 'Wall Oven',
             }

def split_by_day(df, fn_filter=None):
    if fn_filter is None:
        def fn_filter(df):
            return True
    return [group[1] for group in df.groupby(df.index.date) 
            if (fn_filter(group[1]) and len(group[1])==MINUTES_DAY)]


def list_of_series2numpy(list_serie):
    '''
    Dada una lista de series, lo transforma a un numpy de tamano adecuado
    '''
    return np.array([x['current'].to_numpy() for x in list_serie]).T

def get_arrays_per_day(elec_meter):
    '''
    Dado un ElecMeter de nilmtk, devuelve una matriz en la que cada columna
    representa un dia de datos.
    '''
    #fn = lambda x: x['current'].max()>threshold
    df = next(elec_meter.load(resample=True,  sample_period=60))
    
    splitted = split_by_day(df)#, fn)
    return list_of_series2numpy(splitted)

# La función get_D divide la serie de consumo por días y la retorna
def label(x):
    if x.is_site_meter():
        return 'WHE'
    else:
        return x.appliances[0].metadata['original_name']
def get_array_and_label(x):
    return label(x), get_arrays_per_day(x)


def get_df_per_day(path_h5: str) -> pd.DataFrame:
    '''
    Dada la ruta del h5 de AMPds devuelve un dataframe con los datos partidos
    por dia y por medidor.

    Parameters
    ----------
    path_h5 : str
        DESCRIPTION.
    house : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    '''
    
    ds = nilmtk.DataSet(path_h5)
    data_meters = list(map(get_array_and_label, 
                           ds.buildings[1].elec.all_meters()))
    
    # Lo siguiente construye un dataframe multindex como el que se quiere
    n_days = data_meters[0][1].shape[1] #cantidad de dias
    days = pd.RangeIndex(0, n_days, name='sample') #construyo uno de los indices
    names = list(map(lambda x:x[0], data_meters)) #el otro indice correspondiente al nombre
    iterables = [names, days]
    #construccion del multindex
    columns = pd.MultiIndex.from_product(iterables, names=['name', 'day'])
    data = list(map(lambda x:x[1], data_meters))
    data = np.hstack(data)
    df = pd.DataFrame(data, columns=columns)
    df = df.swaplevel(axis=1).sort_index(axis=1)
    return df