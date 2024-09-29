"""


"""
import os

import numpy as np
import torch as th
from matplotlib import pyplot as plt
import joblib as jb

import plotly.graph_objects as go

from load_files import load_files
plt.rc('font', family='Arial')

# load models
model_name_list = os.listdir('trained_models/')
model_list = list()
for name in model_name_list:
    if name[-1] == 'l':
        model_list.append(jb.load(f'trained_models/{name}', ), )
    else:
        model_list.append(th.load(f'trained_models/{name}', ), )

# load files
data = load_files('dataset_sig.xlsx', label_col=-2, )
n_samp, n_feat = data.samp_feat.shape
# expand to interpolate x1
data.samp_feat = np.repeat(data.samp_feat, 50, axis=0)
data.samp_name = np.repeat(data.samp_name, 50, axis=0)
x1 = np.linspace(0, 1, 50)
x1 = np.repeat(x1[None, :], n_samp, axis=0)
data.samp_feat[:, -1] = x1.flatten()
# classified by components
component_set = {data.samp_name[0, 1:].tobytes(), }
split_indices = list()
for i, name in enumerate(data.samp_name):
    hashName = name[1:].tobytes()
    if hashName not in component_set:
        component_set.add(hashName)
        split_indices.append(i)
# split by component
pred_name_list = np.split(data.samp_name, split_indices, axis=0)
pred_data_list = np.split(data.samp_feat, split_indices, axis=0)

plt.figure(figsize=(7, 7))
fig = go.Figure()
system_name_list = list()
for i, pred_name in enumerate(pred_name_list):
    # prediction
    system_name = f'{pred_name[0, 1]} - {pred_name[0, 2]}'
    system_name_list.append(system_name)
    pred_feat = pred_data_list[i]
    predictions_T = model_list[2].predict(pred_feat)  # MLP_T: 1, RF_T: 2, GBR_T: 3
    predictions_y = model_list[5].predict(pred_feat)  # MLP_T: 0, RF_y: 5, GBR_y: 4

    # plot
    #plt.subplot(10, 10, i+1)
    x = pred_feat[:, -1]
    y = predictions_T
    z = predictions_y
    '''
    plt.plot(x, y, 'ro-')
    plt.plot(z, y, 'ro-')
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.xlabel('$x_1$($y_1$)', fontdict={'fontsize': 22, })
    plt.ylabel('$T$/K', fontdict={'fontsize': 22, })
    plt.title(system_name, fontdict={'fontsize': 26, }, y=1.02)
    plt.savefig(f'Figs/{system_name}.jpg', dpi=600)
    plt.clf()'''

    # Plotly Show
    fig.add_trace(
        go.Scatter(
            visible=False,
            mode = 'lines+markers',
            line=dict(color="red", width=2),
            name='Bubble Points Line',
            x=x,
            y=y,
            legend='legend1'
        )
    )
    fig.add_trace(
        go.Scatter(
            visible=False,
            mode = 'lines+markers',
            line=dict(color="blue", width=2),
            name='Drew Points Line',
            x=z,
            y=y,
            legend='legend1'
        )
    )
    fig.layout.legend1 = dict(x=0,y=1.09,#设置图例的位置，[0,1]之间
        font=dict(family='Arial',size=36,color='black'),#设置图例的字体及颜色
        bgcolor='#E2E2E2',bordercolor='#FFFFFF')
steps = []

fig.layout.update(
    updatemenus=[
        go.layout.Updatemenu(
            type = "dropdown", direction = "down",
            buttons=list(
                [
                    dict(
                        args = [{"visible": [False]*(2*i)+[True]*2+[False]*2*(len(system_name_list)-i-1)},{"title": "Sine"}],
                        label = f"{_name}",
                        method = "restyle",

                    ) for i, _name in enumerate(system_name_list)
                ]
            ),
            pad = {"r": 2, "t": 2},
            showactive = True,
            x = 0.72,
            xanchor = "left",
            y = 1.05,
            yanchor = "top",
            font=dict(family='Arial',size=36, color='black')
        ),
]
                      )
fig.show()