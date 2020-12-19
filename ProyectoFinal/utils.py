# -*- coding: utf-8 -*-

MINUTES_DAY = 60*24


import numpy as np


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

def get_D(elec_meter, threshold=1e-5):
    '''
    Dado un ElecMeter de nilmtk y un umbral para no tener vectores nulos, 
    devuelve  el vector D que resuelve el problema de optimzacion
    '''
    fn = lambda x: x['current'].max()>threshold
    df = next(elec_meter.load(resample=True,  sample_period=60))
    
    splitted = split_by_day(df, fn)
    return list_of_series2numpy(splitted)

#%%
# Creacion de U
def get_U(k, d):
    U = np.ones((k, d))
    return U

#%%
# Creacion de Q
def get_Q(k, T, ki):
    if len(ki)-1!=k:
        raise ValueError('len(ki)-1 es distinto de k')
    Q = np.zeros((k, T))
    for i in range(len(ki)-1):
        Q[i, ki[i]:ki[i+1]] = 1
    return Q

def get_G(X, U, betta):
    return np.vstack((X, betta*U))

def get_H(D, Q, betta):
    return np.vstack((D, betta*Q))



#%%
def PCEC(X_agg, X_individual) -> float:
    return float(X_individual.sum()/X_agg.sum())