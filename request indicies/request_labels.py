# SP21 DP Team 05
import torch
import torch.nn as nn

import pandas as pd
import numpy as np

print("Loading CSV...")
prob = pd.read_csv('prob.csv', encoding = "UTF-8")

print("Converting to Tensor...")
prob_tensor = torch.tensor(prob.to_numpy())

torch.sum(prob_tensor,1)

from torch.distributions import Categorical

entropy = Categorical(probs = prob_tensor[:,1:]).entropy()

ids = prob_tensor[:,[0]]

print(ids.shape)

y = torch.unsqueeze(entropy,1)
print(y.shape)

out = torch.cat((ids,y),1)

df = pd.DataFrame(out.numpy())

ds = df.sort_values(1,ascending=False)

d = ds.head(12800)[0]

outs = pd.to_numeric(d, downcast='integer')

outs2 = outs.astype(str) + '.png'
outs2.to_csv('request_05.csv',index=False,header=False,line_terminator = ',\n')
