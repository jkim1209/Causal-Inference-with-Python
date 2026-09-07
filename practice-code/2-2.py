#%%
import pandas as pd
import numpy as np

data = pd.read_csv('./data/online_classroom.csv')
data.head()

#%%
df = data.assign(class_format =  
                # df.assign: 원본 프레임 바꾸지 않고 class_format이라는 새로운 컬럼 추가
             np.select([data["format_ol"].astype(bool), data["format_blended"].astype(bool)],
                # np.select: 다중 조건에 따라 값을 선택
            ["Online", "Blended"],
                # format_ol이 TRUE이면 "Online" 부여
                # format_blended가 TRUE이면 "Blended" 부여
            default="face-to-face"
                # 나머지 경우에는 "face-to-face" 부여
             ))

df.groupby(["class_format"]).mean()
# 위 df를 class_format별로 그룹화
# 각 그룹별 평균값 계산

#%%
X = ["gender", "asian", "black", "hawaiian", "hispanic", "unknown", "white"]

mu = df.groupby("class_format")[X].mean()
var = df.groupby("class_format")[X].var()

# print(mu.head())
# print(var.head())

norm_diff = ((mu - mu.loc["face-to-face"]) / np.sqrt((var + var.loc["face-to-face"])/2))
print(norm_diff)
