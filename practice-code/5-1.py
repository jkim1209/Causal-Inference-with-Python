#%%
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv("data/management_training.csv")
df.head()

#%%
smf.old("engagement_score ~ intervention", data=df).fit().summary().tables[1]

#%%
model = smf.ols("""engagement_score ~ intervention + tenure + last_engagement_score
                + department_score + n_of_reports + C(gender) + C(role)""", data = df).fit()

print("ATE: ", model.params["intervention"])
print("95% CI: ", model.conf_int().loc["intervention", :].values.T)

#%%
ps_model = smf.logit("""intervention ~ tenure + last_engagement_score
                    + department_score + n_of_reports + C(gender) + C(role)""", data = df).fit(disp=0)

data_ps = df.assign(
    propensity_score = ps_model.predict(df)
)

data_ps[["intervention", "engagement_score", "propensity_score"]].head()

#%%
model = smf.ols("engagement_score ~ intervention + propensity_score",
                data=data_ps).fit()
model.params["intervention"]

#%%
from sklearn.neighbors import KNeighborsRegressor

T = "intervention"
X = "propensity_score"
Y = "engagement_score"

