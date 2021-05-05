# %%
from tqdm.auto import tqdm
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sys
import zipfile
import pandas as pd
import datatable as dt
import numpy as np
from utils import *

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = '/media/scao/Data/bms_data/'
sys.path.append(HOME)
'''bash
kaggle competitions list --category featured
'''
# %%

if __name__ == '__main__':
    try:
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files('bms-molecular-translation',
                                       path=DATA_DIR, quiet=False)
        data_file = find_files('zip', DATA_DIR)[0]
        with zipfile.ZipFile(data_file, "r") as f:
            # for item in tqdm(f.infolist(), desc='Extracting '):
            #     try:
            #         f.extract(item, DATA_DIR)
            #     except zipfile.error as e:
            #         pass
            f.extractall(DATA_DIR)
    except RuntimeError as err:
        print(f"Needs API token: {err}")
# %%
