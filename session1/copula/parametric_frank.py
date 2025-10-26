import numpy as np
from scipy.optimize import minimize_scalar

def frank_copula_mle(u, v, n):
    # 2) Frank copula log-likelihood (stable form)
    def frank_loglik(theta):
        theta = float(theta)
        if abs(theta) < 1e-10:
            return 0.0  # independence limit: log c = 0
        em1_theta = np.expm1(-theta)           # e^{-θ} - 1
        # D = 1 - e^{-θ} - (1 - e^{-θu})(1 - e^{-θv})
        Du = np.expm1(-theta * u)              # e^{-θu} - 1
        Dv = np.expm1(-theta * v)              # e^{-θv} - 1
        D  = -(em1_theta + Du * Dv)            # stable equivalent
        # log c(u,v) = log|θ| + log|1 - e^{-θ}| - θ(u+v) - 2 log|D|
        logc = (np.log(abs(theta))
                + np.log(abs(-em1_theta))
                - theta * (u + v)
                - 2.0 * np.log(np.abs(D)))
        return np.sum(logc)

    # 3) Maximize log-likelihood over θ (exclude 0)
    # Search on positive and negative ranges; take the best (and compare with θ=0)
    res_pos = minimize_scalar(lambda t: -frank_loglik(t), bounds=(1e-6, 50.0), method="bounded")
    res_neg = minimize_scalar(lambda t: -frank_loglik(t), bounds=(-50.0, -1e-6), method="bounded")

    cand = [
        (0.0, 0.0),  # independence
        (res_pos.x, -res_pos.fun) if res_pos.success else (None, -np.inf),
        (res_neg.x, -res_neg.fun) if res_neg.success else (None, -np.inf),
    ]
    theta_hat, ll = max(cand, key=lambda kv: kv[1])

    # 4) Information criteria (k = 1 parameter: θ)
    k = 1
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "theta": theta_hat,
        "loglik": ll,
        "AIC": aic,
        "BIC": bic,
    }