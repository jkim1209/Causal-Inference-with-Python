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
from sklearn.linear_model import LinearRegression

def doubly_robust(df, formula, T, Y):
    X = dmatrix(formula, df)

    ps_model = LogisticRegression(l1_ratio=0, max_iter=1000).fit(X, df[T])
    ps = ps_model.predict_proba(X)[:, 1]

    m0 = LinearRegression().fit(X[df[T]==0, :], df.query(f"{T}==0")[Y])
    m1 = LinearRegression().fit(X[df[T]==1, :], df.query(f"{T}==1")[Y])

    m0_hat = m0.predict(X)
    m1_hat = m1.predict(X)

    return(
        np.mean(df[T]*(df[Y] - m1_hat)/ps + m1_hat) - np.mean((1-df[T])*(df[Y] - m0_hat)/(1-ps) + m0_hat)
    )

#%%
formula = """tenure + last_engagement_score + department_score 
            + n_of_reports + C(gender) + C(role)"""
T = "intervention"
Y = "engagement_score"

print("DR ATE: ", doubly_robust(df, formula, T, Y))

est_fn = partial(doubly_robust, formula=formula, T=T, Y=Y)
print("95% CI:", bootstrap(df, est_fn))

#%%
# 처치 모델링이 쉬운 경우
# P(T|X)는 쉬우나, E[Y_t|X]는 어려운 경우
np.random.seed(123)

n = 10000
x = np.random.beta(1,1,n).round(2)*2
e = 1/(1+np.exp(-(1+1.5*x)))
t = np.random.binomial(1,e)

y1 = 1
y0 = 1 - 1*x**3
y = t*y1 + (1-t)*y0 + np.random.normal(0, 1, n)

df_easy_t = pd.DataFrame(dict(y=y, x=x, t=t))

print("True ATE: ", np.mean(y1-y0))

# 회귀모델은 True ATE를 맞추지 못함
m0 = smf.ols("y~x", data=df_easy_t.query("t==0")).fit()
m1 = smf.ols("y~x", data=df_easy_t.query("t==1")).fit()
regr_ate = (m1.predict(df_easy_t) - m0.predict(df_easy_t)).mean()

print("Rgression ATE: ", regr_ate)

# 물론 데이터 생성과정을 안다면 회귀모델에 반영하여 정확한 값을 구할 수 있음 (하지만 현실에서는 쉽지 않음)
m = smf.ols("y~t*(x + np.power(x,3))", data=df_easy_t).fit()
regr_ate = (m.predict(df_easy_t.assign(t=1)) - m.predict(df_easy_t.assign(t=0))).mean()

print("Regression ATE: ", regr_ate)

# 처치 모델링하기가 쉬우므로 IPW는 잘 적용될 것임
est_fn = partial(est_ate_with_ps, ps_formula="x", T="t", Y="y")
print("Propensity Score ATE: ", est_fn(df_easy_t))
print("95% CI: ", bootstrap(df_easy_t, est_fn))

# 이중강건 모형
est_fn = partial(doubly_robust, formula="x", T="t", Y="y")
print("DR ATE: ", est_fn(df_easy_t))
print("95% CI: ", bootstrap(df_easy_t, est_fn))

#%% 
# 결과 모델링이 쉬운 경우
# E[Y_t|X]는 쉬우나, P(T|X)는 어려운 경우
np.random.seed(123)

n = 10000
x = np.random.beta(1,1,n).round(2)*2
e = 1/(1+np.exp(-(2*x-x**3)))
t = np.random.binomial(1,e)

y1 = x
y0 = y1 + 1     # 실제 ATE는 -1
y = t*y1 + (1-t)*y0 + np.random.normal(0, 1, n)

df_easy_y = pd.DataFrame(dict(y=y, x=x, t=t))

print("True ATE: ", np.mean(y1-y0))

# 성향점수 모델링이 상대적으로 복잡하므로, IPW는 실제 ATE를 제대로 추정하기 어려움
est_fn = partial(est_ate_with_ps, ps_formula="x", T="t", Y="y")
print("Propensoty Score ATE: ", est_fn(df_easy_y))
print("95% CI: ", bootstrap(df_easy_y, est_fn))

# 하지만 회귀모델은 정확히 ATE를 추정 가능
m0 = smf.ols("y~x", data=df_easy_y.query("t==0")).fit()
m1 = smf.ols("y~x", data=df_easy_y.query("t==1")).fit()
regr_ate = (m1.predict(df_easy_y) - m0.predict(df_easy_y)).mean()

print("Rgression ATE: ", regr_ate)

# 이중강건 모형
est_fn = partial(doubly_robust, formula="x", T="t", Y="y")
print("DR ATE: ", est_fn(df_easy_y))
print("95% CI: ", bootstrap(df_easy_y, est_fn))

#%%
# 연속형 처치에서의 일반화 성향점수
df_cont_t = pd.read_csv("./data/interest_rate.csv")

print(df_cont_t.head())

# 목표는 금리와 상환기간 사이의 관계가 편향되지 않도록 ml_1과 ml_2 보정
m_naive = smf.ols("duration ~ interest", data=df_cont_t).fit()
print(m_naive.summary().tables[1])

#%%
model_t = smf.ols("interest ~ ml_1 + ml_2", data=df_cont_t).fit()

def conditionial_density(x, mean, std):
    denom = std*np.sqrt(2*np.pi)
    num = np.exp(-((1/2)*((x-mean)/std)**2))
    return (num/denom).to_numpy()

gps = conditionial_density(df_cont_t["interest"], model_t.fittedvalues, np.std(model_t.resid))

print(gps)

#%%
# 정규함수는 패키지로부터 불러올 수도 있음
from scipy.stats import norm

gps = norm(loc=model_t.fittedvalues, scale=np.std(model_t.resid)).pdf(df_cont_t["interest"])

print(gps)

#%%
# # 참고: 처치가 정규분포가 아닌 다른 분포를 따른다면 일반화선형모델(GLS)를 사용하여 적합시킬 수 있음
# import statsmodels.api as sm
# from scipy.stats import poisson

# mt = smf.glm("t~x1+x2", data=df, family=sm.families.Poisson()).fit()

# gps = poisson(mu=m_pois.fittedvalues).pmf(df["t"])

# w = 1/gps

#%%
final_model = smf.wls("duration~interest", data=df_cont_t, weights=1/gps).fit()

final_model.params["interest"]

#%%
# 가중화 안정치는 연속형에서 매우 중요
stabilizer = norm(
    loc=df_cont_t["interest"].mean(),
    scale=np.std(df_cont_t["interest"] - df_cont_t["interest"].mean())
).pdf(df_cont_t["interest"])

gipw = stabilizer/gps

print("Original Sample Size: ", len(df_cont_t))
print("Effective Stable Weights Sample Size: ", sum(gipw))

final_model = smf.wls("duration~interest", data=df_cont_t, weights=gipw).fit()

print(final_model.params["interest"])

#%%
def gps_ate(df, ps_formula, T, Y, stable=False):
    df = df.reset_index(drop=True)  # 중복 인덱스로 인한 pandas 정렬 오류 방지

    model_t = smf.ols(f"{T} ~ {ps_formula}", data=df).fit()
    gps = norm(loc=model_t.fittedvalues, scale=np.std(model_t.resid)).pdf(df[T])

    if stable:
        stabilizer = norm(
            loc=df[T].mean(),
            scale=np.std(df[T] - df[T].mean())
        ).pdf(df[T])
        weights = stabilizer / gps
    else:
        weights = 1 / gps

    final_model = smf.wls(f"{Y} ~ {T}", data=df, weights=weights).fit()
    return final_model.params[T]

#%%
ps_formula = "ml_1 + ml_2"
T = "interest"
Y = "duration"

est_fn_nonstable = partial(gps_ate, ps_formula=ps_formula, T=T, Y=Y, stable=False)
est_fn_stable = partial(gps_ate, ps_formula=ps_formula, T=T, Y=Y, stable=True)

print("Point estimate, non-stable: ", est_fn_nonstable(df_cont_t))
print("95% CI, non-stable: ", bootstrap(df_cont_t, est_fn_nonstable))

print("Point estimate, stable: ", est_fn_stable(df_cont_t))
print("95% CI, stable: ", bootstrap(df_cont_t, est_fn_stable))