## 블록 1: 데이터 불러오기

```python
df = pd.read_csv("data/management_training.csv")
df.head()
```

- **1행**: `pd.read_csv(...)`로 CSV 파일을 읽어서 데이터프레임 `df`에 저장. `management_training.csv`는 "관리자 교육(intervention)을 받은 직원과 안 받은 직원의 이후 engagement_score"를 담은 데이터로 추정됩니다.
- **2행**: `.head()`는 상위 5개 행을 미리보기로 출력 (컬럼 이름, 데이터 형태를 확인하는 용도).

문제의식: "교육을 받은 직원이 실제로 더 높은 engagement를 보이는가, 아니면 애초에 교육 대상으로 선택된 직원들의 특성(승진 앞둔 사람, 저성과자 등) 때문에 그렇게 보이는 것뿐인가?"

------

## 블록 2: Naive 비교 (편향의 출발점)

```python
smf.ols("engagement_score ~ intervention", data=df).fit().summary().tables[1]
```

- **`smf.ols("engagement_score ~ intervention", data=df)`**: `engagement_score`를 종속변수, `intervention`을 유일한 독립변수로 하는 OLS(최소제곱회귀) 모형을 정의. `formula.api`의 `~` 문법은 R 스타일 수식.
- **`.fit()`**: 실제로 회귀계수를 추정 (모형 적합).
- **`.summary()`**: 회귀결과 전체 요약(계수, 표준오차, R², F통계량 등)을 담은 객체 반환.
- **`.tables[1]`**: summary 안에는 표가 여러 개 있는데(보통 [0]=모형 전체 정보, [1]=계수표, [2]=잔차진단), 인덱스 1번인 **계수표**(coef, std err, t, P>|t|, 95% CI)만 뽑아서 보는 것.

**🔍 왜 이 계수가 "두 집단의 평균 차이"와 같은지**

`intervention`은 0 또는 1만 갖는 더미변수입니다. 단순선형회귀 $Y = \beta_0 + \beta_1 T$에서, $T=0$일 때 예측값은 $\beta_0$, $T=1$일 때 예측값은 $\beta_0+\beta_1$입니다. OLS는 각 집단 내에서 예측오차 제곱합을 최소화하도록 적합되므로, 결과적으로 $\beta_0 = \bar{Y}_{T=0}$(비교군 평균), $\beta_0+\beta_1 = \bar{Y}_{T=1}$(처치군 평균)이 됩니다. 따라서:
$$
\beta_1 = \bar{Y}_{T=1} - \bar{Y}_{T=0}
$$

즉 회귀계수 $\beta_1$이 곧 두 집단의 평균 차이입니다.

이게 왜 문제냐면, 이 데이터가 무작위실험(RCT)이 아니라 관찰데이터이기 때문이에요. 관리자가 "성과가 낮은 사람"을 골라 교육에 보냈다면, `intervention=1`인 사람들은 애초에 `last_engagement_score`가 낮은 사람들일 테니, 이 계수는 교육의 효과와 "원래 그런 사람이었다는 효과(selection bias)"가 뒤섞인 값입니다.

------

## 블록 3: 공변량을 직접 통제하는 회귀 (Regression Adjustment)

```python
model = smf.ols("""engagement_score ~ intervention + tenure + last_engagement_score
                + department_score + n_of_reports + C(gender) + C(role)""", data = df).fit()

print("ATE: ", model.params["intervention"])
print("95% CI: ", model.conf_int().loc["intervention", :].values.T)
```

- **1행**: 여러 줄 문자열(`"""..."""`)로 수식을 작성. `engagement_score`를 `intervention` 외에도 `tenure`, `last_engagement_score`, `department_score`, `n_of_reports`, `C(gender)`, `C(role)`로 설명하는 다중회귀. `C(...)`는 해당 변수를 범주형(categorical)으로 처리해 자동으로 더미변수화하라는 patsy 문법. `.fit()`으로 적합.
- **2행**: `model.params`는 모든 계수를 담은 pandas Series. `["intervention"]`으로 인덱싱해서 `intervention` 계수만 꺼내 출력. 이 계수가 다른 공변량을 통제한 뒤의 ATE 추정치.
- **3행**: `model.conf_int()`는 모든 파라미터의 95% 신뢰구간을 담은 데이터프레임(행=파라미터, 열=[하한, 상한]) 반환. `.loc["intervention", :]`로 `intervention` 행만 선택(두 값짜리 Series). `.values`로 numpy 배열 변환. `.T`(전치)는 1차원 배열에는 사실상 아무 효과가 없지만(값 순서 그대로), 습관적으로 붙인 것으로 보입니다.

**🔍 이 방법이 기대는 가정**

$$
 (Y_1, Y_0) \perp T \mid X
$$

"관측된 공변량 X(tenure, 이전 engagement, 부서점수, 부하직원수, 성별, 직급)를 다 통제하고 나면, 남은 처치 여부는 잠재적 결과와 독립적이다"는 조건부 무교란성(unconfoundedness) 가정입니다. 이게 성립한다면 `intervention`의 계수가 ATE(평균처치효과)로 해석됩니다.

약점: 선형함수형태를 가정하고, X가 많아지면 차원의 저주 문제가 생기고, 관측 안 된 교란변수가 있으면 여전히 편향됩니다.

------

## 블록 4: 성향점수 모델링

```python
ps_model = smf.logit("""intervention ~ tenure + last_engagement_score
                    + department_score + n_of_reports + C(gender) + C(role)""", data = df).fit(disp=0)

data_ps = df.assign(
    propensity_score = ps_model.predict(df)
)

data_ps[["intervention", "engagement_score", "propensity_score"]].head()
```

- **1행**: `smf.logit(...)`으로 `intervention`(0/1)을 종속변수로, 나머지 공변량들을 독립변수로 하는 **로지스틱 회귀**를 정의. `.fit(disp=0)`에서 `disp=0`은 최적화 반복 과정의 로그 출력을 끄는 옵션(다른 의미 없음, 그냥 출력이 지저분해지지 않게 하는 용도).
- **2~4행**: `df.assign(propensity_score = ps_model.predict(df))`는 원본 `df`에 `propensity_score`라는 새 컬럼을 추가한 **새 데이터프레임**을 만들어 `data_ps`에 저장(`assign`은 원본을 바꾸지 않고 복사본을 반환). `ps_model.predict(df)`는 로짓모형이 계산한 "각 사람이 처치를 받았을 확률" 값들.
- **5행**: `intervention`, `engagement_score`, `propensity_score` 세 컬럼만 골라서 상위 5행 확인.

**🔍 balancing score 정리**

로지스틱 회귀로 추정하는 값은:

$$
 e(X) = P(T=1 \mid X)
$$

Rosenbaum & Rubin (1983)의 정리에 따르면 이 $e(X)$는 **balancing score**입니다:

$$
 X \perp T \mid e(X)
$$

즉 "PS가 같은 사람들끼리는 X의 분포가 같다." 그리고 unconfoundedness가 X에서 성립하면 e(X)에서도 성립합니다:

$$
 (Y_1, Y_0) \perp T \mid X \implies (Y_1, Y_0) \perp T \mid e(X)
$$

→ X 전체를 통제하는 것과 e(X) 하나만 통제하는 것이 이론적으로 동치가 됩니다.

------

## 블록 5: PS를 직접 회귀에 넣기

```python
model = smf.ols("engagement_score ~ intervention + propensity_score",
                data=data_ps).fit()
model.params["intervention"]
```

- **1~2행**: `engagement_score`를 `intervention`과 `propensity_score` 두 개만으로 설명하는 OLS 모형을 적합. 앞서 6개 넘는 공변량을 전부 나열했던 블록 3과 달리, 여기서는 그걸 압축한 `propensity_score` 하나만 통제변수로 씁니다.
- **3행**: `intervention` 계수만 꺼내서 확인.

X를 전부 나열하는 대신 e(X) 하나로 압축해서 통제하는 방식. 이론적으로는 블록 3과 동등하지만, PS 모형이 정확히 맞아야 하고 선형항 하나로는 e(X) 내부의 비선형 관계를 완전히 못 잡는다는 근사적 한계가 있습니다.

------

## 블록 6: K-최근접이웃 매칭 (PSM)

```python
from sklearn.neighbors import KNeighborsRegressor

T = "intervention"
X = "propensity_score"
Y = "engagement_score"
```

- **1행**: sklearn에서 K-최근접이웃 회귀 클래스를 불러옴.
- **2~4행**: 이후 코드에서 반복적으로 쓸 컬럼 이름들을 문자열 변수로 저장(가독성 및 재사용을 위한 관례).

```python
# 처치군과 대조군 데이터 분리
treated = data_ps.query(f"{T}==1")
untreated = data_ps.query(f"{T}==0")
```

- **1행**: `data_ps.query(f"{T}==1")`은 `data_ps.query("intervention==1")`과 같은 뜻(f-string으로 변수를 수식 문자열에 끼워넣음). 처치군만 골라 `treated`에 저장.
- **2행**: 같은 방식으로 비교군만 골라 `untreated`에 저장.

```python
mt0 = KNeighborsRegressor(n_neighbors=1).fit(untreated[[X]], untreated[Y])
mt1 = KNeighborsRegressor(n_neighbors=1).fit(treated[[X]], treated[Y])
```

- **1행**: `KNeighborsRegressor(n_neighbors=1)`은 "가장 가까운 이웃 1개의 Y값을 그대로 예측값으로 쓰는" 모형(사실상 최근접 매칭). `.fit(untreated[[X]], untreated[Y])`로 **비교군 안에서** "PS(X) → engagement_score(Y)" 관계를 학습. `untreated[[X]]`는 대괄호를 두 번 써서 (1열짜리) 데이터프레임 형태를 유지(sklearn이 2차원 입력을 요구하기 때문).
- **2행**: 같은 방식으로 **처치군 안에서** 학습한 모형 `mt1`.

```python
predicted = pd.concat([
    # 대조건 knn 모델을 통해 실험군의 짝을 찾기
    treated.assign(match=mt0.predict(treated[[X]])),

    # 실험군 knn 모델을 통해 대조군의 짝을 찾기
    untreated.assign(match=mt1.predict(untreated[[X]]))
])

predicted.head()
```

- **1~5행**: `treated.assign(match=mt0.predict(treated[[X]]))`는 처치군 각 사람의 PS값을 **비교군에서 학습한 모형 `mt0`**에 넣어서, "이 사람과 PS가 가장 비슷한 비교군 사람의 Y값"을 `match`라는 새 컬럼으로 붙임 → 처치군 각자의 반사실 $Y_0$의 대리값. `untreated.assign(match=mt1.predict(untreated[[X]]))`는 반대로 비교군 각 사람에게 **처치군에서 학습한 모형 `mt1`**을 적용해 반사실 $Y_1$의 대리값을 붙임. `pd.concat([...])`으로 두 결과를 위아래로 합침.
- **6행**: 합쳐진 결과 상위 5행 확인.

```python
# ATE
np.mean((predicted[Y] - predicted["match"])*predicted[T] + 
        (predicted["match"] - predicted[Y])*(1-predicted[T]))
```

- **`(predicted[Y] - predicted["match"])\*predicted[T]`**: `predicted[T]`가 1(처치군)인 행에서만 살아남는 항. 처치군의 (실제 관측 Y) − (매칭된 반사실 $Y_0$) = 그 사람의 개별처치효과 추정치.
- **`(predicted["match"] - predicted[Y])\*(1-predicted[T])`**: `predicted[T]`가 0(비교군)인 행에서만 살아남는 항. 비교군의 (매칭된 반사실 $Y_1$) − (실제 관측 Y) = 역시 그 사람의 개별처치효과 추정치.
- **`np.mean(... + ...)`**: 두 항을 더한 뒤 전체 평균 → 모든 사람 각자의 개별처치효과를 하나로 모아 ATE 추정.

이게 매칭(matching) 기법입니다: "PS가 balancing score이므로, PS가 비슷한 두 사람은 서로의 반사실 대용품이 될 수 있다"는 논리를 그대로 구현한 것.

------

## 블록 7: IPW를 직접 계산 (가중치 만들기)

```python
weight_t = 1/data_ps.query("intervention==1")["propensity_score"]
weight_nt = 1/(1-data_ps.query("intervention==0")["propensity_score"])
```

- **1행**: 처치군만 골라 `propensity_score` 컬럼을 가져온 뒤 역수를 취함. 처치군 각 사람마다 "1 ÷ 자기 성향점수" 가중치.
- **2행**: 비교군에 대해, "1 − 성향점수"(=처치를 안 받을 확률)의 역수를 취함.

**왜 이런 식일까?** 처치군 사람의 가중치는 $1/e(X)$, 비교군 사람의 가중치는 $1/(1-e(X))$. 처치군 입장에선 $e(X)$가 "이 사람이 처치를 받을 확률"이니 그 역수를, 비교군 입장에선 $1-e(X)$가 "이 사람이 비교군에 속할 확률"이니 그 역수를 쓰는 대칭 구조.

```python
t1 = data_ps.query("intervention==1")["engagement_score"]
t0 = data_ps.query("intervention==0")["engagement_score"]
```

- **1행**: 처치군의 `engagement_score`(결과변수 Y)만 뽑아 `t1`에 저장.
- **2행**: 비교군의 `engagement_score`를 `t0`에 저장.

```python
y1 = sum(t1*weight_t)/len(data_ps)
y0 = sum(t0*weight_nt)/len(data_ps)
```

- **1행**: `t1*weight_t`는 처치군 각 사람의 (Y값 × 자기 가중치)를 원소별로 곱한 것. 다 더하고(`sum`) 전체 표본 크기(`len(data_ps)`, 처치군+비교군 합친 전체 N)로 나눔 → $\hat{E}[Y_1]$.
- **2행**: 같은 방식으로 $\hat{E}[Y_0]$.

**🔍 왜 이 계산이 $E[Y_1]$을 추정하는지**

$$
E[Y_1] = E\left[\frac{T \cdot Y}{e(X)}\right]
$$

$T$가 0/1이므로 $T=0$인 사람은 항 전체가 0이 되어 기여하지 않고, $T=1$인 사람만 $Y/e(X)$로 남습니다. 조건부기댓값 반복법칙을 쓰면:

$$
E\left[\frac{T \cdot Y}{e(X)}\right] = E\left[E\left[\frac{T \cdot Y}{e(X)} \,\Big|\, X\right]\right] = E\left[\frac{e(X)\cdot E[Y\mid X, T=1]}{e(X)}\right] = E[E[Y\mid X, T=1]] = E[Y_1]
$$

(두 번째 등호: $X$가 주어지면 $e(X)$는 상수이고, $T\cdot Y$는 $T=1$일 때만 $Y$로 남으므로 $E[T\cdot Y\mid X]=e(X)\cdot E[Y\mid X,T=1]$. 이걸 $e(X)$로 나누면 네 번째 항이 됨. unconfoundedness 가정 하에 $E[Y\mid X,T=1] = E[Y_1\mid X]$이므로 마지막 등호가 성립.)

$e(X)$로 나눠주는 게 "이 사람이 뽑힐 확률이 낮았던 만큼 더 크게 쳐준다"는 보정 역할을 해서, 표본평균이 모집단 평균의 불편추정량이 되게 합니다.

```python
print("E[Y1]: ", y1)
print("E[Y0]: ", y0)
print("ATE: ", y1-y0)
```

- 계산된 $\hat{E}[Y_1]$, $\hat{E}[Y_0]$, 그 차이(ATE 추정치)를 출력.

------

## 블록 8: 압축된 한 줄 IPW 공식

```python
np.mean(data_ps["engagement_score"]
        * (data_ps["intervention"] - data_ps["propensity_score"])
        / (data_ps["propensity_score"]*(1-data_ps["propensity_score"])))
```

- 블록 7을 처치군/비교군으로 나누지 않고 전체 데이터에 한 번에 계산하는 버전. `engagement_score` × (T − e(X)) ÷ (e(X)(1−e(X)))의 전체 평균.

**🔍 블록 7과 같은 값이 나오는 이유**

$T=1$일 때 $\dfrac{Y(1-e(X))}{e(X)(1-e(X))} = \dfrac{Y}{e(X)}$, $T=0$일 때 $\dfrac{Y(0-e(X))}{e(X)(1-e(X))} = \dfrac{-Y}{1-e(X)}$이므로

$$
\frac{T-e(X)}{e(X)(1-e(X))}\cdot Y = \frac{T}{e(X)}\cdot Y - \frac{1-T}{1-e(X)}\cdot Y
$$

즉 처치군에서는 $Y/e(X)$를 더하고, 비교군에서는 $Y/(1-e(X))$를 뺀다는 뜻 — 블록 7의 $\hat{E}[Y_1]-\hat{E}[Y_0]$와 동일한 계산.

------

## 블록 9: `est_ate_with_ps` 함수 정의

```python
from sklearn.linear_model import LogisticRegression
from patsy import dmatrix
```

- **1행**: sklearn 로지스틱 회귀 클래스.
- **2행**: 수식 문자열을 디자인 행렬로 바꿔주는 patsy의 `dmatrix`.

```python
def est_ate_with_ps(df, ps_formula, T, Y):
    X = dmatrix(ps_formula, df)
    ps_model = LogisticRegression(l1_ratio=0, max_iter=1000).fit(X, df[T])
    ps = ps_model.predict_proba(X)[:, 1]
    return np.mean((df[T]-ps) / (ps*(1-ps)) * df[Y])
```

- **함수 정의**: `df`, 성향점수 모형용 수식 `ps_formula`, 처치변수명 `T`, 결과변수명 `Y`를 받음.
- **`X = dmatrix(ps_formula, df)`**: 문자열 수식을 실제 행렬로 변환(`C()`는 자동 더미화).
- **`ps_model = LogisticRegression(...).fit(X, df[T])`**: 로지스틱 회귀 학습. `l1_ratio=0`은 기본 penalty(`l2`)에서는 실제로 무시되는 파라미터. `max_iter=1000`은 수렴 반복 횟수를 넉넉히 설정.
- **`ps = ps_model.predict_proba(X)[:, 1]`**: `predict_proba`는 [P(T=0), P(T=1)] 두 열을 반환하는데, 두 번째 열(P(T=1)=성향점수)만 추출.
- **`return np.mean(...)`**: 블록 8의 압축 공식 그대로 적용해 ATE 하나를 반환.

```python
formula = """tenure + last_engagement_score 
            + department_score + n_of_reports + C(gender) + C(role)"""
T = "intervention"
Y = "engagement_score"

est_ate_with_ps(df, formula, T, Y)
```

- 공변량 수식 문자열 정의, T·Y 지정, 함수를 원본 `df`에 적용해서 ATE 하나를 바로 계산.

------

## 블록 10: `bootstrap` 함수 정의 및 실행

```python
from joblib import Parallel, delayed
from toolz import partial
```

- **1행**: 병렬처리 도구.
- **2행**: 함수 일부 인자를 미리 고정해 새 함수를 만드는 `partial`.

```python
def bootstrap(data, est_fn, rounds=200, seed=123, pcts=[2.5, 97.5]):
    np.random.seed(seed)

    stats = Parallel(n_jobs=4)(
        delayed(est_fn)(data.sample(frac=1, replace=True))
        for _ in range(rounds)
    )

    return np.percentile(stats, pcts)
```

- **함수 정의**: `data`, ATE 계산 함수 `est_fn`, 반복횟수 `rounds`(기본 200), 시드 `seed`(기본 123), 퍼센타일 구간 `pcts`(기본 [2.5, 97.5]).
- **`np.random.seed(seed)`**: 재현성 확보.
- **`Parallel(n_jobs=4)(...)`**: 4개 코어로 병렬 실행.
- **`delayed(est_fn)(data.sample(frac=1, replace=True))`**: `data.sample(frac=1, replace=True)`는 복원추출로 원본 크기만큼 재표집한 부트스트랩 표본 1개. `delayed(est_fn)(...)`는 지연 실행 객체.
- **`for _ in range(rounds)`**: 200번 반복 → ATE 추정치 200개.
- **`return np.percentile(stats, pcts)`**: 2.5%, 97.5% 위치 값을 신뢰구간 경계로 사용.

```python
print(f"ATE: {est_ate_with_ps(df, formula, T, Y)}")

est_fn = partial(est_ate_with_ps, ps_formula = formula, T=T, Y=Y)

print(f"95% CI: ", bootstrap(df, est_fn))
```

- 원본 데이터로 점추정치 출력.
- `partial`로 `ps_formula, T, Y`를 고정해 `df` 하나만 받으면 되는 함수 `est_fn` 생성.
- `bootstrap(df, est_fn)`으로 95% 신뢰구간 출력.

------

## 블록 11: Pseudo-population 크기 점검

```python
print("Original Sample Size: ", data_ps.shape[0])
print("Treated Pseudo-Population Sample Size: ", sum(weight_t))
print("Untreated Pseudo-Population Sample Size: ", sum(weight_nt))
```

- **1행**: 원본 행 개수(전체 N).
- **2행**: 처치군 가중치 합(`1/e(X)`) — "전체 모집단이 다 처치받았다고 가정했을 때의 가상 표본 크기".
- **3행**: 비교군 가중치 합.

가중치 합이 원본 N과 크게 다르면(특히 훨씬 크면) 극단적 가중치가 섞여 있다는 신호 — positivity 위반과 직결됩니다.

------

## 블록 12: 안정화 가중치(Stabilized weights) 계산

```python
p_of_t = data_ps["intervention"].mean()
```

- 전체 표본에서 `intervention=1` 비율, $P(T=1)$의 표본추정치.

```python
t1 = data_ps.query("intervention==1")
t0 = data_ps.query("intervention==0")
```

- 처치군/비교군 전체 행(모든 컬럼)을 각각 저장. (블록 7의 `t1`은 컬럼 하나였는데, 여기서는 행 전체를 담은 데이터프레임으로 변수명이 재활용됨.)

```python
weight_t_stable = p_of_t / t1["propensity_score"]
weight_nt_stable = (1-p_of_t) / (1-t0["propensity_score"])
```

- 처치군: $P(T=1)/e(X)$. 비교군: $(1-P(T=1))/(1-e(X))$.

**🔍 왜 안정화되는지**

$E[e(X)] \approx P(T=1)$이므로 $\dfrac{P(T=1)}{e(X)}$는 평균적으로 1 근처. 반면 원래 가중치 $1/e(X)$는 $e(X)$가 0에 가까우면 폭발적으로 커질 수 있음. 불편성은 유지하면서 분산만 줄이는 기법.

```python
print("Treat size:", len(t1))
print("W treat", sum(weight_t_stable))

print("Control size:", len(t0))
print("W treat", sum(weight_nt_stable))
```

- 각 그룹 크기와 안정화 가중치 합 출력.

------

## 블록 13: 안정화 가중치로 ATE 계산

```python
nt = len(t1)
nc = len(t0)

y1 = sum(t1["engagement_score"]*weight_t_stable)/nt
y0 = sum(t0["engagement_score"]*weight_nt_stable)/nc

print("ATE: ", y1-y0)
```

- 처치군/비교군 크기 저장. 각 그룹의 (Y×가중치) 합을 **그 그룹 크기**로 나눔(블록 7과 달리 분모가 전체 N이 아님 — 안정화 가중치가 이미 그룹 크기 근처로 스케일되어 있기 때문). 차이 = ATE.

------

## 블록 14: Positivity(양수성) 위반 시뮬레이션 데이터 생성

```python
np.random.seed(1)

n = 1000
x = np.random.normal(0, 1, n)
t = np.random.normal(x, 0.5, n) > 0
```

- **시드 고정, n=1000**.
- **`x`**: 표준정규분포에서 생성한 공변량(교란변수).
- **`t`**: 평균이 `x`, 표준편차 0.5인 정규분포 난수가 0보다 크면 True(처치). x가 클수록 t=True 확률 높음. x가 극단값이면 t가 거의 결정적으로 정해짐 → positivity 위반의 핵심 장치.

```python
y0 = -x
y1 = y0 + t     # 실제 ATE = 1
```

- $Y_0 = -x$, $Y_1 = Y_0 + t$(t가 bool→int로 캐스팅되어 사실상 +1) → $Y_1-Y_0=1$, 진짜 ATE는 1.

```python
y = np.random.normal((1-t)*y0 + t*y1, 0.2)
```

- t=0이면 y0, t=1이면 y1이 선택되는 식으로 실제 관측 Y를 구성하고, 표준편차 0.2 정규분포 노이즈를 섞음.

```python
df_no_pos = pd.DataFrame(dict(x=x, t=t.astype(int), y=y))
df_no_pos.head()
```

- x, t(정수 변환), y를 컬럼으로 하는 데이터프레임 생성 및 미리보기.

------

## 블록 15: Positivity 위반 상황에서 IPW 시도

```python
est_fn = partial(est_ate_with_ps, ps_formula="x", T="t", Y="y")
print("ATE: ", est_fn(df_no_pos))
print(f"95% C.I.: ", bootstrap(df_no_pos, est_fn))
```

- PS모형에 x 하나만 넣어 함수 고정. 점추정치와 95% 신뢰구간 출력.
- 신뢰구간 상한이 실제 ATE(=1)보다 낮게 나옴 — positivity 위반으로 IPW가 편향된다는 증거.

------

## 블록 16: 단순 회귀와 비교

```python
smf.ols("y ~ x + t", data = df_no_pos).fit().params["t"]
```

- x와 t를 둘 다 넣은 선형회귀에서 t의 계수(처치효과 추정치) 확인.
- 운 좋게 1에 가깝게 나오지만, 데이터가 거의 없는 x 영역까지 회귀직선을 외삽한 결과라 신뢰하기 어렵다는 경고.

------

## 블록 17: 이중강건(Doubly Robust) 함수 정의

```python
from sklearn.linear_model import LinearRegression
```

- 결과모형에 쓸 선형회귀 클래스.

```python
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
```

- `X`, `ps_model`, `ps`: 블록 9와 동일한 성향점수 계산.
- **`m0 = LinearRegression().fit(X[df[T]==0, :], df.query(f"{T}==0")[Y])`**: `X[df[T]==0, :]`는 비교군에 해당하는 행만 불리언 인덱싱으로 선택(넘파이 배열이라 `.query`가 아닌 불리언 마스크). 그 행들만으로 "X→Y" 선형회귀 학습 → 비교군용 결과모형 $m_0$.
- **`m1 = LinearRegression().fit(...)`**: 같은 방식으로 처치군용 결과모형 $m_1$.
- **`m0_hat = m0.predict(X)`**, **`m1_hat = m1.predict(X)`**: 각 모형을 **전체 표본**에 적용해 "비교군/처치군이었다면 Y가 얼마였을까"를 모두에게 예측.
- **`return(...)`**: 아래 도출 참고.

**🔍 반환식 도출 및 의미**

$$
 \hat{E}[Y_1] = \frac{1}{N}\sum_i \left[\frac{T_i(Y_i - \hat{m}_1(X_i))}{e(X_i)} + \hat{m}_1(X_i)\right]
$$

- $\hat{m}_1(X_i)$: 결과모형의 예측값을 베이스라인으로 깔아둠(모두에게).
- $\dfrac{T_i(Y_i - \hat{m}_1(X_i))}{e(X_i)}$: 처치군만 살아남는 항. (실제값−예측값=잔차)를 IPW 방식으로 보정해 더함.

$\hat{E}[Y_0]$도 대칭적으로 계산, 두 값의 차이가 ATE.

**왜 이중으로 강건한가**: 결과모형이 맞으면 잔차항 기댓값이 0이 되어 PS모형이 틀려도 문제 없음. PS모형이 맞으면 이 식이 순수 IPW와 같아져 결과모형이 틀려도 문제 없음. 둘 중 하나만 맞아도 일치추정량.

------

## 블록 18: DR을 원본 데이터에 적용 + 부트스트랩

```python
formula = """tenure + last_engagement_score + department_score 
            + n_of_reports + C(gender) + C(role)"""
T = "intervention"
Y = "engagement_score"

print("DR ATE: ", doubly_robust(df, formula, T, Y))

est_fn = partial(doubly_robust, formula=formula, T=T, Y=Y)
print("95% CI:", bootstrap(df, est_fn))
```

- 공변량 수식 재정의. `doubly_robust`를 원본 데이터에 적용해 DR ATE 출력. `partial`로 고정한 함수를 `bootstrap`에 넘겨 95% 신뢰구간 계산.

------

## 블록 19: "처치 모델링이 쉬운 경우" 시뮬레이션

```python
np.random.seed(123)

n = 10000
x = np.random.beta(1,1,n).round(2)*2
e = 1/(1+np.exp(-(1+1.5*x)))
t = np.random.binomial(1,e)
```

- 시드, n=10000. `x`: Beta(1,1)(≈[0,1] 균등분포)에서 뽑아 반올림 후 2배 → [0,2] 구간. `e`: 로지스틱 함수로 x의 **선형결합**(1+1.5x)을 확률로 변환 — 진짜 성향점수가 "로짓모형으로 정확히 잡히는" 단순 구조. `t`: 확률 `e`로 베르누이 시행.

```python
y1 = 1
y0 = 1 - 1*x**3
y = t*y1 + (1-t)*y0 + np.random.normal(0, 1, n)
```

- $Y_1=1$(상수). $Y_0 = 1-x^3$(3차함수 — "결과모형이 어렵다"는 부분). Y는 t에 따라 선택 후 노이즈 추가.

```python
df_easy_t = pd.DataFrame(dict(y=y, x=x, t=t))
print("True ATE: ", np.mean(y1-y0))
```

- 데이터프레임 생성. 실제 ATE = $E[x^3]$의 표본평균 출력.

```python
m0 = smf.ols("y~x", data=df_easy_t.query("t==0")).fit()
m1 = smf.ols("y~x", data=df_easy_t.query("t==1")).fit()
regr_ate = (m1.predict(df_easy_t) - m0.predict(df_easy_t)).mean()

print("Rgression ATE: ", regr_ate)
```

- 비교군만/처치군만으로 각각 "y~x" 선형회귀 학습, 전체 데이터에 예측해 차이의 평균 = ATE 추정. x-y0 관계가 3차함수인데 선형으로 잡으려니 편향될 것.

```python
m = smf.ols("y~t*(x + np.power(x,3))", data=df_easy_t).fit()
regr_ate = (m.predict(df_easy_t.assign(t=1)) - m.predict(df_easy_t.assign(t=0))).mean()

print("Regression ATE: ", regr_ate)
```

- `"y~t*(x + np.power(x,3))"`: t와 (x, x³)의 교호작용 포함 — patsy에서 `t + x + x^3 + t:x + t:x^3`로 전개. 진짜 데이터생성과정(3차항)을 알고 넣은 모형. `assign(t=1)`/`assign(t=0)`으로 전체 표본의 t를 강제로 바꿔 예측한 뒤 차이 평균 = ATE. 현실에서는 불가능한 "정답을 아는" 대조군 실험.

```python
est_fn = partial(est_ate_with_ps, ps_formula="x", T="t", Y="y")
print("Propensity Score ATE: ", est_fn(df_easy_t))
print("95% CI: ", bootstrap(df_easy_t, est_fn))
```

- PS모형에 x 하나만 넣고 IPW 추정. PS모형(로짓+x)이 실제 처치배정 구조와 일치하니 잘 맞을 것으로 예상.

```python
est_fn = partial(doubly_robust, formula="x", T="t", Y="y")
print("DR ATE: ", est_fn(df_easy_t))
print("95% CI: ", bootstrap(df_easy_t, est_fn))
```

- 같은 데이터에 DR도 적용해 비교.

------

## 블록 20: "결과 모델링이 쉬운 경우" 시뮬레이션

```python
np.random.seed(123)

n = 10000
x = np.random.beta(1,1,n).round(2)*2
e = 1/(1+np.exp(-(2*x-x**3)))
t = np.random.binomial(1,e)
```

- 시드, n, x는 앞과 동일. `e`: 성향점수 함수 안에 **x³ 항 포함**된 복잡한 구조 — "처치모형이 어렵다"는 부분. t는 그 확률로 베르누이 시행.

```python
y1 = x
y0 = y1 + 1     # 실제 ATE는 -1
y = t*y1 + (1-t)*y0 + np.random.normal(0, 1, n)
```

- $Y_1=x$(선형 — "결과모형이 쉽다"). $Y_0=Y_1+1$ → $Y_1-Y_0=-1$이 정확한 ATE. Y 관측치 생성은 동일 구조.

```python
df_easy_y = pd.DataFrame(dict(y=y, x=x, t=t))
print("True ATE: ", np.mean(y1-y0))
```

- 데이터프레임 생성 및 진짜 ATE(-1) 출력.

```python
est_fn = partial(est_ate_with_ps, ps_formula="x", T="t", Y="y")
print("Propensoty Score ATE: ", est_fn(df_easy_y))
print("95% CI: ", bootstrap(df_easy_y, est_fn))
```

- PS모형에 x만 선형으로 넣어 IPW 추정. 진짜 $e(X)$는 x³ 항이 필요한데 이 모형은 그걸 못 잡으니 편향될 것으로 예상.

```python
# 하지만 회귀모델은 정확히 ATE를 추정 가능
m0 = smf.ols("y~x", data=df_easy_y.query("t==0")).fit()
m1 = smf.ols("y~x", data=df_easy_y.query("t==1")).fit()
regr_ate = (m1.predict(df_easy_y) - m0.predict(df_easy_y)).mean()

print("Rgression ATE: ", regr_ate)
```

  - 비교군(`t==0`)과 처치군(`t==1`) 각각 `df_easy_y`로 "y~x" 선형회귀를 학습(`m0`, `m1`).
  - 학습에 쓴 것과 같은 데이터 `df_easy_y` 전체에 두 모형을 예측시켜, 그 차이의 평균을 ATE로 계산.
  - 이 시뮬레이션은 $Y_1=x$, $Y_0=x+1$로 설계되어 있어 x-Y 관계가 완전히 선형이므로, 선형회귀 함수형태가 정확히 들어맞습니다. 그래서 `regr_ate`는 실제 ATE인 -1에 가깝게 나와야 합니다 — "결과모형이 쉬우면 단순회귀만으로도 정확한 ATE를 잡을 수 있다"는 이 블록의 취지가 이제 제대로 확인됩니다.

```python
est_fn = partial(doubly_robust, formula="x", T="t", Y="y")
print("DR ATE: ", est_fn(df_easy_y))
print("95% CI: ", bootstrap(df_easy_y, est_fn))
```

- `df_easy_y`에 DR 적용. 결과모형이 쉬우므로 DR도 정답에 가까울 것으로 예상.

------

## 블록 21: 연속형 처치 데이터 불러오기 + naive 회귀

```python
df_cont_t = pd.read_csv("./data/interest_rate.csv")
print(df_cont_t.head())
```

- CSV 로드(대출 금리 `interest`, 상환기간 `duration`, 신용평가 스코어 `ml_1`, `ml_2` 등으로 추정). 상위 5행 확인.

```python
m_naive = smf.ols("duration ~ interest", data=df_cont_t).fit()
print(m_naive.summary().tables[1])
```

- 교란변수 없이 `duration`을 `interest` 하나로만 설명하는 단순회귀 적합, 계수표 출력. 교란변수(ml_1, ml_2)를 무시한 편향 가능성 있는 베이스라인.

------

## 블록 22: GPS(일반화 성향점수) 직접 계산

```python
model_t = smf.ols("interest ~ ml_1 + ml_2", data=df_cont_t).fit()
```

- 처치변수(interest, 연속형)를 공변량(ml_1, ml_2)으로 설명하는 선형회귀. 이산형의 로짓모형 역할을 대신하는 처치모형. `fittedvalues`는 "이 공변량 조건에서 예상되는 금리의 평균".

```python
def conditionial_density(x, mean, std):
    denom = std*np.sqrt(2*np.pi)
    num = np.exp(-((1/2)*((x-mean)/std)**2))
    return (num/denom).to_numpy()
```

- 정규분포 PDF를 직접 구현. `denom = std*np.sqrt(2*np.pi)`는 $\sigma\sqrt{2\pi}$. `num = np.exp(...)`은 $\exp\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$. 둘을 나눠 PDF값 반환, `.to_numpy()`로 numpy 배열 변환.

**🔍 정규분포 PDF의 출처**

$$
 f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

"평균 $\mu$에서 멀어질수록 지수적으로 확률밀도가 줄어드는" 종모양 곡선을 수식화한 통계학의 표준 정의를 그대로 코드로 옮긴 것.

```python
gps = conditionial_density(df_cont_t["interest"], model_t.fittedvalues, np.std(model_t.resid))
print(gps)
```

- 실제 `interest` 값을, 각자의 예측된 평균(`fittedvalues`, 사람마다 다름)과 잔차의 표준편차(`np.std(model_t.resid)`, 전체 공통) 기준으로 밀도 계산 → 각 사람의 GPS. 출력 확인.

------

## 블록 23: scipy로 동일 계산

```python
from scipy.stats import norm

gps = norm(loc=model_t.fittedvalues, scale=np.std(model_t.resid)).pdf(df_cont_t["interest"])
print(gps)
```

- `scipy.stats.norm(loc=평균, scale=표준편차)`로 정규분포 객체 생성, `.pdf(값)`으로 밀도 계산. 블록 22와 수학적으로 동일한 계산을 라이브러리로 대체.

------

## 블록 24: (주석) 다른 분포를 쓰는 경우 참고

```python
# # 참고: 처치가 정규분포가 아닌 다른 분포를 따른다면 일반화선형모델(GLS)를 사용하여 적합시킬 수 있음
# import statsmodels.api as sm
# from scipy.stats import poisson

# mt = smf.glm("t~x1+x2", data=df, family=sm.families.Poisson()).fit()

# gps = poisson(mu=m_pois.fittedvalues).pmf(df["t"])

# w = 1/gps
```

- 전부 주석 처리되어 실행되지 않음. 처치변수가 카운트 데이터라면 `smf.glm(..., family=Poisson())`으로 포아송 회귀를 적합하고, `poisson(mu=예측값).pmf(관측값)`으로 확률질량함수(PMF)를 GPS로 쓸 수 있다는 메모.

------

## 블록 25: GPS로 가중회귀 (WLS)

```python
final_model = smf.wls("duration~interest", data=df_cont_t, weights=1/gps).fit()
final_model.params["interest"]
```

- `smf.wls(...)`: 가중최소제곱회귀. `weights=1/gps`로 GPS의 역수를 가중치로 부여 — 밀도가 낮은(흔치 않은 금리 값) 사람일수록 가중치가 커짐. `.params["interest"]`로 처치효과 추정치 확인.

------

## 블록 26: 연속형 안정화 가중치

```python
stabilizer = norm(
    loc=df_cont_t["interest"].mean(),
    scale=np.std(df_cont_t["interest"] - df_cont_t["interest"].mean())
).pdf(df_cont_t["interest"])
```

- `loc`: 공변량 조건 없이 interest 전체 평균. `scale`: interest 전체 표준편차(`np.std`가 자동으로 평균을 빼는 기능을 하므로, 명시적으로 평균을 빼는 건 결과에 영향 없이 같은 값을 줌 — 가독성 목적). `.pdf(...)`: 각 관측치가 "공변량과 무관한 전체 분포" 기준으로 얼마나 흔한지 밀도 계산 → 한계밀도 $f(t)$.

```python
gipw = stabilizer/gps
```

- $\dfrac{f(t)}{f(t\mid X)}$ 형태의 안정화 가중치. 이산형의 $P(T=1)/e(X)$와 대응.

```python
print("Original Sample Size: ", len(df_cont_t))
print("Effective Stable Weights Sample Size: ", sum(gipw))
```

- 원본 표본 크기와 안정화 가중치 합 비교 — 가중치 안정성 진단.

```python
final_model = smf.wls("duration~interest", data=df_cont_t, weights=gipw).fit()
print(final_model.params["interest"])
```

- 안정화 가중치로 재적합, 계수 출력.

------

## 블록 27: `gps_ate` 함수로 통합 (부트스트랩용)

```python
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
```

- **`df = df.reset_index(drop=True)`**: 부트스트랩 복원추출로 생기는 중복 인덱스를 재설정 — Series 간 연산의 인덱스 정렬 오류 방지.
- **`model_t = smf.ols(f"{T} ~ {ps_formula}", data=df).fit()`**: f-string으로 처치변수명과 공변량 수식을 동적으로 끼워넣어 처치모형 적합.
- **`gps = norm(...).pdf(df[T])`**: 블록 22~23과 동일한 GPS 계산.
- **`if stable: ... else: ...`**: `stable=True`면 안정화 가중치(블록 26), `False`면 `1/gps`(블록 25) — 두 버전을 함수 하나로 통합.
- **`final_model = smf.wls(f"{Y} ~ {T}", data=df, weights=weights).fit()`**: 가중회귀 적합.
- **`return final_model.params[T]`**: 처치변수 계수(ATE)만 반환.

```python
ps_formula = "ml_1 + ml_2"
T = "interest"
Y = "duration"

est_fn_nonstable = partial(gps_ate, ps_formula=ps_formula, T=T, Y=Y, stable=False)
est_fn_stable = partial(gps_ate, ps_formula=ps_formula, T=T, Y=Y, stable=True)

print("Point estimate, non-stable: ", est_fn_nonstable(df_cont_t))
print("95% CI, non-stable: ", bootstrap(df_cont_t, est_fn_nonstable))

print("Point estimate, stable: ", est_fn_stable(df_cont_t))
print("95% CI, stable: ", bootstrap(df_cont_t, est_fn_stable))
```

- 공변량 수식, T, Y 정의. `partial`로 non-stable/stable 두 버전의 함수 생성(`df` 하나만 받으면 되도록). 각각 점추정치와 95% 신뢰구간 출력 — 안정화가 신뢰구간(분산)에 미치는 영향을 비교하는 게 마지막 블록의 목적.