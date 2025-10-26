import numpy as np
from scipy.stats import norm

def to_u_v(returns, volume):
    x = np.asarray(returns)
    y = np.asarray(volume)

    # Parametric marginals (Normal) -> PIT to U(0,1)
    mu_x, sd_x = x.mean(), x.std(ddof=1)
    mu_y, sd_y = y.mean(), y.std(ddof=1)
    u = norm.cdf(x, loc=mu_x, scale=sd_x)
    v = norm.cdf(y, loc=mu_y, scale=sd_y)

    # clip to avoid 0/1 which break inverse CDF)
    eps = 1e-12
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    return u, v
