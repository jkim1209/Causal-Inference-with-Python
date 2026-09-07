#%%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

data = pd.read_csv("./data/cross_sell_email.csv")
print(data.head(10))

data.groupby("cross_sell_email").size()

#%%
short_email = data.query("cross_sell_email=='short'")["conversion"]
long_email = data.query("cross_sell_email=='long'")["conversion"]
email = data.query("cross_sell_email=='email'")["conversion"]
no_email = data.query("cross_sell_email=='no_email'")["conversion"]

def se(y):
    return y.std() / np.sqrt(len(y))

print("SE for long email: ", se(long_email))
print("SE for short email: ", se(short_email))

#%%
print("SE for long email: ", long_email.sem())
print("SE for short email: ", short_email.sem())

#%%
n = 100
conv_rate = 0.08

def run_experiment():
    return np.random.binomial(1, conv_rate, size=n)

np.random.seed(42)
experiments = [run_experiment().mean() for _ in range(1000)]

plt.figure(figsize=(8,5))
freq, bins, img = plt.hist(experiments, bins=20, label="Experiment Means")
plt.vlines(conv_rate, ymin=0, ymax=freq.max(), linestyles="dashed", label="True Mean", color="orange")
plt.legend()

# %%
exp_se = short_email.sem()
exp_mu = short_email.mean()
ci95 = (exp_mu - 2 * exp_se, exp_mu + 2 * exp_se)
print("95% CI for short email: ", ci)

x = np.linspace(exp_mu - 4*exp_se, exp_mu + 4*exp_se, 100)
y = stats.norm.pdf(x, exp_mu, exp_se)
plt.plot(x, y)
plt.vlines([ci95[1], ci95[0]], ymin=0, ymax=3, linestyles=':', colors='red', label="95% CI")
plt.legend()
plt.show()

#%%
z = np.abs(stats.norm.ppf((1-.99)/2))
print(z)

ci99 = (exp_mu - z * exp_se, exp_mu + z * exp_se)
ci99

x = np.linspace(exp_mu - 4*exp_se, exp_mu + 4*exp_se, 100)
y = stats.norm.pdf(x, exp_mu, exp_se)
plt.plot(x, y)
plt.vlines([ci95[1], ci95[0]], ymin=0, ymax=3, linestyles=':', colors='red', label="95% CI")
plt.vlines([ci99[1], ci99[0]], ymin=0, ymax=3, linestyles='--', colors='red', label="99% CI")
plt.legend()
plt.show()

#%%
def ci(y):
    return(y.mean() - 2 * y.sem(), y.mean() + 2 * y.sem())

print("95% CI for short email: ", ci(short_email))
print("95% CI for long email: ", ci(long_email))
print("95% CI for no_email: ", ci(no_email))

groups = {
    "short_email": short_email,
    "long_email": long_email,
    "no_email": no_email
}
colors = {"short_email": "blue", "long_email": "red", "no_email": "green"}

for email, y in groups.items():
    mu = y.mean()
    se = y.sem()
    ci95 = ci(y)

    x = np.linspace(mu - 4*se, mu + 4*se, 100)
    y_pdf = stats.norm.pdf(x, mu, se)

    plt.plot(x, y_pdf, color=colors[email], label=f"{email} (mean={mu: .2f})")
    plt.vlines([ci95[1], ci95[0]], ymin=0, ymax=stats.norm.pdf(ci95, mu, se), linestyles=':', colors=colors[email])
    plt.fill_between(x, y_pdf, where=(x >= ci95[0]) & (x <= ci95[1]), color=colors[email], alpha=0.2)

plt.xlabel("Conversion Rate")
plt.legend()
plt.show()

#%%
np.random.seed(123)

n1 = np.random.normal(4,3,30000)
n2 = np.random.normal(1,4,30000)
n_diff = n2 - n1

plt.figure(figsize=(10,4))
sns.distplot(n1, hist=False, label="$N(4,3^2)$")
sns.distplot(n2, hist=False, label="$N(1,4^2)$")
sns.distplot(n_diff, hist=False, label="$N(-3,5^2) = N(1,4^2) - N(4,3^2)$")
plt.legend()
plt.show()

#%%
diff_mu = short_email.mean() - no_email.mean()
diff_se = np.sqrt(no_email.sem()**2 + short_email.sem()**2)

ci = (diff_mu - 1.96 * diff_se, diff_mu + 1.96 * diff_se)
print(f"95% CI for the difference (short email - no email): \n{ci}")

x = np.linspace(diff_mu - 4*diff_se, diff_mu + 4*diff_se, 100)
y = stats.norm.pdf(x, diff_mu, diff_se)
plt.plot(x, y)
plt.vlines([ci[1], ci[0]], ymin=0, ymax=3, linestyles=':', colors='red', label="95% CI")
plt.legend()
plt.show()

#%%
diff_mu_shifted = short_email.mean() - no_email.mean() - 0.01
diff_se = np.sqrt(no_email.sem()**2 + short_email.sem()**2)

ci = (diff_mu_shifted - 1.96 * diff_se, diff_mu_shifted + 1.96 * diff_se)
print(f"95% CI for the difference (short email - no email): \n{ci}")

x = np.linspace(diff_mu_shifted - 4*diff_se, diff_mu_shifted + 4*diff_se, 100)
y = stats.norm.pdf(x, diff_mu_shifted, diff_se)
plt.plot(x, y)
plt.vlines([ci[1], ci[0]], ymin=0, ymax=3, linestyles=':', colors='red', label="95% CI")
plt.legend()
plt.show()

#%%
t_stat = (diff_mu - 0) / diff_se
t_stat

#%%
np.ceil(16*(no_email.std()/0.08)**2)

data.groupby("cross_sell_email").size()
# %%
