import numpy as np
from scipy.stats import norm

def gaussian_copula_mle(u, v, n):
    # Transform to Gaussian scores
    z1 = norm.ppf(u)
    z2 = norm.ppf(v)

    # MLE for Gaussian copula correlation
    rho = np.corrcoef(z1, z2)[0, 1]
    rho = np.clip(rho, -0.999999, 0.999999)  # numerical safety

    # Log-likelihood of Gaussian copula
    one_minus_r2 = 1.0 - rho**2
    ll_terms = (
        -0.5 * np.log(one_minus_r2)
        - ( -2 * rho * z1 * z2 + (rho**2) * (z1**2 + z2**2) ) / (2 * one_minus_r2)
    )
    ll = np.sum(ll_terms)  # total log-likelihood

    # Information criteria (k=1 parameter: rho)
    k = 1
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "rho": rho,
        "loglik": ll,
        "AIC": aic,
        "BIC": bic
    }
