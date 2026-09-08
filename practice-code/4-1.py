#%%
import pandas as pd
import numpy as np

data = pd.read_csv("./data/rec_ab_test.csv")
data.head()

#%%
import statsmodels.formula.api as smf

result = smf.ols('watch_time ~ C(recommender)', data=data).fit()    # C 는 범주형임을 나타냄
result.summary().tables[1]

#%%
(data
.groupby("recommender")
         ["watch_time"]
         .mean())

#%%
risk_data = pd.read_csv("./data/risk_data.csv")
risk_data.head()

#%%
model = smf.ols('default ~ credit_limit', data=risk_data).fit()
model.summary().tables[1]

#%%
print(risk_data.groupby(["credit_score1", "credit_score2"]).size().head())

#%%
formula = 'default ~ credit_limit + wage + credit_score1 + credit_score2'
model = smf.ols(formula, data=risk_data).fit()
model.summary().tables[1]

#%%
X_cols = ["credit_limit", "wage", "credit_score1", "credit_score2"]
X = risk_data[X_cols].assign(intercep=1)
y = risk_data["default"]

def regress(y,X):
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

beta = regress(y,X)
beta

#%%
debiasing_model = smf.ols(
    'credit_limit ~ wage + credit_score1 + credit_score2',
    data = risk_data
).fit()

risk_data_deb = risk_data.assign(
    # 시각화를 위해, avg(T)를 잔차에 추가
    credit_limit_res = (debiasing_model.resid + risk_data["credit_limit"].mean())
)

model_w_deb_data = smf.ols('default ~ credit_limit_res', data = risk_data_deb).fit()
model_w_deb_data.summary().tables[1]

#%%
denoising_model = smf.ols(
    'default ~ wage + credit_score1 + credit_score2',
    data = risk_data_deb
).fit()

rist_data_denoise = risk_data_deb.assign(
    defualt_res = denoising_model.resid + risk_data_deb["default"].mean()
)

#%%
model_se = smf.ols(
    'default ~ wage + credit_score1 + credit_score2',
    data = risk_data
).fit()

print("SE regression: ", model_se.bse["wage"])

model_wage_aux = smf.ols(
    "wage ~ credit_score1 + credit_score2",
    data = risk_data
).fit()

se_formula = (np.std(model_se.resid) 
              / (np.std(model_wage_aux.resid)*np.sqrt(len(risk_data)-4)))

print("SE formula: ", se_formula)
