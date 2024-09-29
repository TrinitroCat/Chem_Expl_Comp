"""


"""
import os

import numpy as np
import torch as th
from matplotlib import pyplot as plt
import joblib as jb

from load_files import load_files

import shap

# load models
model_name_list = os.listdir('trained_models/')
model_list = list()
for name in model_name_list:
    if name[-1] == 'l':
        model_list.append(jb.load(f'trained_models/{name}', ), )
    else:
        #model_list.append(th.load(f'trained_models/{name}', ), )
        pass

print(model_list)
# load files
data = load_files('dataset.xlsx', label_col=-2, )
n_samp, n_feat = data.samp_feat.shape

explainer = shap.Explainer(model_list[3], feature_names=data.feat_name, algorithm='tree')
shap_values = explainer(data.samp_feat)
shap.summary_plot(shap_values, data.samp_feat, show=False)
plt.savefig('SHAP_y.tiff', dpi=600)
