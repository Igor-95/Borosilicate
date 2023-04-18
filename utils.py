import pandas as pd
import numpy as np
import os
import glob


def get_cwd():
    return  os.getcwd()

def get_all_files(wd=False):
    if wd == False:
        cwd = get_cwd()
    else:
        cwd = wd
    all_files = glob.glob(os.path.join(cwd, "*"))
    return all_files

def baseline_correct_spectra(data):
    df = pd.DataFrame()
    df = data
    y = np.array(df.iloc[:,1])
    y = y - np.min(y)
    df.iloc[:,1] = y
    return df

def normalize_bc_spectra(data):
    df = baseline_correct_spectra(data)
    y = df.iloc[:,1]
    integral = np.trapz(y, df.iloc[:,0])
    y = np.array(y)/integral
    df.iloc[:,1] = y
    return df


