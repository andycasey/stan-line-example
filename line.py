import stan_utils as stan
import numpy as np
import matplotlib.pyplot as plt
from corner import corner

np.random.seed(42)

m_true, b_true = truths = np.random.uniform(-1, 5, size=2)

N = 42

x_err_intrinsic = 5
x_err = np.abs(np.random.normal(0, x_err_intrinsic, size=N))

y_err_intrinsic = 20
y_err = np.abs(np.random.normal(0, y_err_intrinsic, size=N))

x_true = np.random.uniform(0, 30, N)
x = x_true + np.random.normal(0, 1, size=N) * x_err

y_true = m_true * x + b_true
y = y_true + np.random.normal(0, 1, size=N) * y_err


fig, ax = plt.subplots()
ax.scatter(x, y)
ax.errorbar(x, y, yerr=y_err)



data_dict = dict(N=N, x=x, y=y, y_err=y_err, x_err=x_err)
init_dict = dict(m=0, b=0, x_true=x)

sm = stan.load_stan_model("advanced_line.stan")

opt = sm.optimizing(data=data_dict)

sampling = sm.sampling(**stan.sampling_kwds(init=init_dict, data=data_dict, chains=2, iter=1000))

chains = sampling.extract()

X = np.array([chains["m"], chains["b"]]).T

fig = corner(X, truths=truths)
plt.show()