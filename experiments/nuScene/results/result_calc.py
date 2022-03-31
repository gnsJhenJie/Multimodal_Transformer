import pandas as pd

data = pd.read_csv('lane_transformer_me_8_ade_full.csv')
data = data[~data.value.isnull()]
print('Total ADE : ',data['value'].mean())

data = pd.read_csv('lane_transformer_me_8_fde_full.csv')
data = data[~data.value.isnull()]
print('Total FDE : ',data['value'].mean())