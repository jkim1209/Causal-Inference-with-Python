#%%
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv("data/management_training.csv")
df.head()

#%%
smf.ols("engagement_score ~ intervention", data=df).fit().summary().tables[1]

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

# 처치군과 대조군 데이터 분리
treated = data_ps.query(f"{T}==1")
untreated = data_ps.query(f"{T}==0")

mt0 = KNeighborsRegressor(n_neighbors=1).fit(untreated[[X]], untreated[Y])
mt1 = KNeighborsRegressor(n_neighbors=1).fit(treated[[X]], treated[Y])

predicted = pd.concat([
    # 대조건 knn 모델을 통해 실험군의 짝을 찾기
    treated.assign(match=mt0.predict(treated[[X]])),

    # 실험군 knn 모델을 통해 대조군의 짝을 찾기
    untreated.assign(match=mt1.predict(untreated[[X]]))
])

predicted.head()

#%%
# ATE
np.mean((predicted[Y] - predicted["match"])*predicted[T] + 
        (predicted["match"] - predicted[Y])*(1-predicted[T]))

#%%
# ATE formula
weight_t = 1/data_ps.query("intervention==1")["propensity_score"]
weight_nt = 1/(1-data_ps.query("intervention==0")["propensity_score"])

t1 = data_ps.query("intervention==1")["engagement_score"]
t0 = data_ps.query("intervention==0")["engagement_score"]

y1 = sum(t1*weight_t)/len(data_ps)
y0 = sum(t0*weight_nt)/len(data_ps)

print("E[Y1]: ", y1)
print("E[Y0]: ", y0)
print("ATE: ", y1-y0)

#%%
# ATE can be also calculated by...
np.mean(data_ps["engagement_score"]
        * (data_ps["intervention"] - data_ps["propensity_score"])
        / (data_ps["propensity_score"]*(1-data_ps["propensity_score"])))

#%%
from sklearn.linear_model import LogisticRegression
from patsy import dmatrix

# IPW 추정량을 계산하는 함수 정의
def est_ate_with_ps(df, ps_formula, T, Y):
    X = dmatrix(ps_formula, df)
    ps_model = LogisticRegression(l1_ratio=0, max_iter=1000).fit(X, df[T])
    ps = ps_model.predict_proba(X)[:, 1]
    # ATE 계산
    return np.mean((df[T]-ps) / (ps*(1-ps)) * df[Y])

formula = """tenure + last_engagement_score 
            + department_score + n_of_reports + C(gender) + C(role)"""
T = "intervention"
Y = "engagement_score"

est_ate_with_ps(df, formula, T, Y)

#%%
from joblib import Parallel, delayed
from toolz import partial

def bootstrap(data, est_fn, rounds=200, seed=123, pcts=[2.5, 97.5]):
    np.random.seed(seed)

    stats = Parallel(n_jobs=4)(
        delayed(est_fn)(data.sample(frac=1, replace=True))
        # Parallel(n_jobs=4) + delayed(...): 이 200번의 계산을 CPU 4개를 
        # 써서 동시에(병렬로) 처리 → 속도 향상
        # data.sample(frac=1, replace=True): 원본 데이터에서 복원추출(중복 허용)로 
        # 같은 크기(100%)만큼 랜덤 샘플링 → 이게 "부트스트랩 샘플" 하나.
        for _ in range(rounds)
    )

    return np.percentile(stats, pcts)

print(f"ATE: {est_ate_with_ps(df, formula, T, Y)}")

est_fn = partial(est_ate_with_ps, ps_formula = formula, T=T, Y=Y)

print(f"95% CI: ", bootstrap(df, est_fn))

#%%
print("Original Sample Size: ", data_ps.shape[0])
print("Treated Pseudo-Population Sample Size: ", sum(weight_t))
print("Untreated Pseudo-Population Sample Size: ", sum(weight_nt))

#%%
p_of_t = data_ps["intervention"].mean()

t1 = data_ps.query("intervention==1")
t0 = data_ps.query("intervention==0")

weight_t_stable = p_of_t / t1["propensity_score"]
weight_nt_stable = (1-p_of_t) / (1-t0["propensity_score"])

print("Treat size:", len(t1))
print("W treat", sum(weight_t_stable))

print("Treat size:", len(t0))
print("W treat", sum(weight_nt_stable))

#%%
nt = len(t1)
nc = len(t0)

y1 = sum(t1["engagement_score"]*weight_t_stable)/nt
y0 = sum(t0["engagement_score"]*weight_nt_stable)/nc

print("ATE: ", y1-y0)

#%%
# 큰 분산과 양수성 가정 위배하는 경우
np.random.seed(1)

n = 1000
x = np.random.normal(0, 1, n)
t = np.random.normal(x, 0.5, n) > 0

y0 = -x
y1 = y0 + t     # 실제 ATE = 1

y = np.random.normal((1-t)*y0 + t*y1, 0.2)

df_no_pos = pd.DataFrame(dict(x=x, t=t.astype(int), y=y))

df_no_pos.head()

#%%
est_fn = partial(est_ate_with_ps, ps_formula="x", T="t", Y="y")
print("ATE: ", est_fn(df_no_pos))
print(f"95% C.I.: ", bootstrap(df_no_pos, est_fn))

# 신뢰구간의 상한이 실제 ATE인 1보다 현저히 낮아보임

#%%
smf.ols("y ~ x + t", data = df_no_pos).fit().params["t"]

# 운이 좋아 1로 나옴, 하지만 회귀분석은 실제 데이터가 전혀 없는 영역까지 외삽한다.

#%%
# 이중 강건 추정
