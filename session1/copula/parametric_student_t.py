import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize
from scipy.special import gammaln

def t_copula_mle(u, v, n):
    # helper: log pdf of univariate t_ν at q
    def log_t_pdf(q, nu):
        return (
            gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - ((nu + 1) / 2) * np.log1p((q * q) / nu)
        )

    # Negative log-likelihood of t-copula
    # Parameterization: a = atanh(ρ), b = log(ν - 2)  ->  ρ in (-1,1), ν > 2
    def neg_ll(params):
        a, b = params
        rho = np.tanh(a)
        nu = 2.0 + np.exp(b)

        q1 = t.ppf(u, df=nu)
        q2 = t.ppf(v, df=nu)

        s = 1.0 - rho * rho
        inv_quad = (q1 * q1 - 2 * rho * q1 * q2 + q2 * q2) / s

        # log pdf of bivariate t_ν with correlation ρ at (q1,q2)
        log_const2 = (
            gammaln((nu + 2) / 2) - gammaln(nu / 2)
            - np.log(nu) - np.log(np.pi) - 0.5 * np.log(s)
        )
        log_f2 = log_const2 - ((nu + 2) / 2) * np.log1p(inv_quad / nu)

        # copula log-density: log c = log f2 - log f1(q1) - log f1(q2)
        log_f1_q1 = log_t_pdf(q1, nu)
        log_f1_q2 = log_t_pdf(q2, nu)
        logc = log_f2 - (log_f1_q1 + log_f1_q2)

        return -np.sum(logc)

    # Initialize and optimize
    z1 = norm.ppf(u)
    z2 = norm.ppf(v)
    r0 = np.clip(np.corrcoef(z1, z2)[0, 1], -0.99, 0.99)
    a0 = np.arctanh(r0)          # initial ρ via Gaussian scores
    b0 = np.log(10.0 - 2.0)      # initial ν ≈ 10

    res = minimize(neg_ll, x0=np.array([a0, b0]), method="L-BFGS-B")
    a_hat, b_hat = res.x
    rho_hat = np.tanh(a_hat)
    nu_hat = 2.0 + np.exp(b_hat)
    ll = -res.fun

    # Information criteria (k = 2: ρ and ν)
    k = 2
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "rho": float(rho_hat),
        "nu": float(nu_hat),
        "loglik": float(ll),
        "AIC": float(aic),
        "BIC": float(bic),
        "converged": bool(res.success),
        "message": res.message,
    }
