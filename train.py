"""

"""
import time
import copy

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib as jb
import torch as th

from ppx_tools.main import Select, Trans

from MLP_model import MLP, TrainMLP

from load_files import load_files

RFx = RandomForestRegressor(
    n_estimators=200,
    max_depth=18,
    min_samples_leaf=2,
    min_samples_split=2,
    verbose=0,
    n_jobs=-1
)
RFy = RandomForestRegressor(
    n_estimators=400,
    max_depth=16,
    min_samples_leaf=1,
    min_samples_split=2,
    verbose=0,
    n_jobs=-1
)

GBRx = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=300,
    verbose=1,
)
GBRy = copy.deepcopy(GBRx)

'''
MLPx = MLPRegressor(
    hidden_layer_sizes=(150,150,150,150),
    solver='adam',
    learning_rate='adaptive',
    learning_rate_init=0.0005,
    batch_size=275,
    max_iter=2000,
    n_iter_no_change=50,
    verbose=True
)
MLPy = copy.deepcopy(MLPx)
'''
# torch implementation
MLPx = MLP(20, 4, 100)
MLPy = MLP(20, 4, 100)

# load files
data = load_files('origin_dataset.xlsx', label_col=-2)
# classified by components
component_set = {data.samp_name[0, 1:].tobytes(), }
split_indices = list()
for i, name in enumerate(data.samp_name):
    hashName = name[1:].tobytes()
    if hashName not in component_set:
        component_set.add(hashName)
        split_indices.append(i)
# split
data.blocked_split(split_indices, True, 0.1, False, 0.1)

# feature engineering
'''trans = Trans(data.train_feat[:, :-1], data.train_label, data.feat_name[:-1])
X_trans, X_trans_name = trans.elem_func(name=True)
data.train_feat = np.concatenate([X_trans, data.train_feat[:, -1:]], axis=1)
data.train_names = np.concatenate([X_trans_name, data.feat_name[-1:]], axis=0)

sel = Select(data.train_feat, data.train_label[:, 0], data.feat_name)
X_select = sel.SIS('distance', size=300)
sel = Select(X_select, data.train_label[:, 0], data.feat_name)
sel.RFE(RFx, 1, )'''

# train
RFx.fit(data.train_feat, data.train_label[:, 0])
RFy.fit(data.train_feat, data.train_label[:, 1])
GBRx.fit(data.train_feat, data.train_label[:, 0])
GBRy.fit(data.train_feat, data.train_label[:, 1])
'''
MLPx.fit(data.train_feat, data.train_label[:, 0])
MLPy.fit(data.train_feat, data.train_label[:, 1])
'''
# torch train
print('\nTorch Training ...')
trainer_x = TrainMLP(MLPx, model_name='T')
trainer_x.run(2000, th.from_numpy(data.train_feat), th.from_numpy(data.train_label[:, 0]), save_thres=4.)
trainer_y = TrainMLP(MLPy, model_name='y')
trainer_y.run(2000, th.from_numpy(data.train_feat), th.from_numpy(data.train_label[:, 1]), save_thres=0.1)
print('Done. \n')

# validation
print('*'*120)
MODEL_NAME = ('RF', 'GBR', 'MLP')
for algo in (RFx, GBRx, MLPx):
    algo.verbose = 0
print('----- Temperature -----')
print('R^2:')
r2_list = list()
for i, algo in enumerate((RFx, GBRx, MLPx)):
    r2 = r2_score(data.val_label[:, 0], algo.predict(data.val_feat))
    r2_list.append(r2)
    print(f'\t{MODEL_NAME[i]:<5}: {r2:<5.3f}')
print('MAE:')
for i, algo in enumerate((RFx, GBRx, MLPx)):
    print(f'\t{MODEL_NAME[i]:<5}: {mean_absolute_error(data.val_label[:, 0], algo.predict(data.val_feat)):<6.4f}')
print('RMSE:')
for i, algo in enumerate((RFx, GBRx, MLPx)):
    rmse = root_mean_squared_error(data.val_label[:, 0], algo.predict(data.val_feat))
    print(f'\t{MODEL_NAME[i]:<5}: {rmse:<6.4f}')
    # Save models
    jb.dump(algo, f'trained_models/{MODEL_NAME[i]}x_RMSE_{rmse:.4f}_r2_{r2_list[i]:.3f}.pkl')
print('----- Y -----')
print('R^2:')
r2_list = list()
for i, algo in enumerate((RFy, GBRy, MLPy)):
    r2 = r2_score(data.val_label[:, 1], algo.predict(data.val_feat))
    r2_list.append(r2)
    print(f'\t{MODEL_NAME[i]:<5}: {r2:<5.3f}')
print('MAE:')
for i, algo in enumerate((RFy, GBRy, MLPy)):
    print(f'\t{MODEL_NAME[i]:<5}: {mean_absolute_error(data.val_label[:, 1], algo.predict(data.val_feat)):<6.4f}')
print('RMSE:')
for i, algo in enumerate((RFy, GBRy, )):
    rmse = root_mean_squared_error(data.val_label[:, 1], algo.predict(data.val_feat))
    print(f'\t{MODEL_NAME[i]:<5}: {rmse:<6.4f}')
    # Save models
    jb.dump(algo, f'trained_models/{MODEL_NAME[i]}y_RMSE_{rmse:.4f}_r2_{r2_list[i]:.3f}.pkl')
