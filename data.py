#%%
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os, sys
from utils import *
import zipfile
import pandas as pd
import datatable as dt
import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = '/media/scao/Data/hpa_data/'
sys.path.append(HOME) 
# %%

if __name__ == '__main__':
    try:
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files('hpa-single-cell-image-classification',
                                        path=DATA_DIR, quiet=False)
        data_file = find_files('zip', DATA_DIR)
        with zipfile.ZipFile(data_file,"r") as f:
            f.extractall(DATA_DIR)
    except RuntimeError as err:
        print(f"Needs API token: {err}")