#%%
import pandas as pd
import numpy as np

df = pd.read_csv('data/enem_scores.csv')
df.sort_values(by="avg_score", ascending=False).head(10)

# %%
