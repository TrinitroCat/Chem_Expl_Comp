import pandas as pd
import numpy as np

while True:
    print('泡点线输入：')
    print('\t请输入环己烷含量(0～1)，用空格分开：')
    a = input()
    print('\t请输入对应的泡点温度(K)，用空格分开：')
    b = input()
    a = np.array(a.split(), dtype=np.float32)
    b = np.array(b.split(), dtype=np.float32)
    if len(a) != len(b):
        print('所输入环己烷含量与泡点温度样本数不相符，请重新输入。')
        continue
    data = np.concatenate([a[:, None], b[:, None]], axis=1)
    print('泡点线输入完成。输入数据如下：\n' + '-'*100 + '\n')
    input_table = pd.DataFrame(data)
    input_table.columns = ['环己烷含量', '泡点温度/K']
    print(input_table)
    print('-'*100 + '\n是否重新输入？(y/n)')
    is_reinput = input()
    if is_reinput != 'y':
        break
