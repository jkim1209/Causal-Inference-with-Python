#%%
import pandas as pd

df = pd.DataFrame(dict(
    profits_prev_6m = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    consultancy = [0, 0, 1, 0, 1, 1],
    profits_next_6m = [1.0, 1.1, 1.2, 5.5, 5.7, 5.7]
))

df

#%%
(df.query("consultancy == 1")["profits_next_6m"].mean()
 - df.query("consultancy == 0")["profits_next_6m"].mean())

#%%
avg_df = (df.
          groupby(["consultancy", "profits_prev_6m"])
          ["profits_next_6m"].mean())

print(avg_df)

print(avg_df.loc[1] - avg_df.loc[0])