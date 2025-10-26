import numpy as np
from scipy import stats
from scipy.optimize import minimize

# ---------- Helpers ----------
def ecdf_u(x):
    """Empirical CDF -> pseudo-uniforms in (0,1)."""
    r = stats.rankdata(x, method="average")
    n = len(x)
    return r / (n + 1.0)

def aic(ll, k):
    return 2*k - 2*ll

def bic(ll, k, n):
    return k*np.log(n) - 2*ll

# ---------- Copula densities ----------
def c_gaussian(u, v, rho):
    """Gaussian copula density; rho in (-1,1)."""
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    # bivariate normal / (product of univariate normals)
    denom = np.sqrt(1 - rho**2)
    quad = (z1**2 - 2*rho*z1*z2 + z2**2) / (2*(1 - rho**2)) - (z1**2 + z2**2)/2
    c = (1/denom) * np.exp(-quad)
    return np.clip(c, 1e-300, np.inf)

def c_clayton(u, v, theta):
    """Clayton copula density; theta > 0."""
    # c(u,v) = (1+theta) * (u*v)^(-(1+theta)) * (u^{-theta}+v^{-theta}-1)^(-(2+1/theta))
    t = theta
    term = (u**(-t) + v**(-t) - 1.0)
    c = (1+t) * (u*v)**(-(1+t)) * term**(-(2 + 1.0/t))
    return np.clip(c, 1e-300, np.inf)

def c_frank(u, v, theta):
    """Frank copula density; theta != 0."""
    t = theta
    e_t  = np.exp(-t)
    e_tu = np.exp(-t*u)
    e_tv = np.exp(-t*v)
    num = t * (e_t - 1.0) * e_tu * e_tv
    den = ( (e_tu - 1.0)*(e_tv - 1.0) + (e_t - 1.0) )**2
    c = num / den
    return np.clip(c, 1e-300, np.inf)

# ---------- Generic MLE wrappers ----------
def mle_copula(u, v, copula="gaussian"):
    """Return dict with params, ll, aic, bic for a 1-parameter copula fit on (u,v)."""
    n = len(u)
    assert n == len(v)

    if copula == "gaussian":
        # parameter: rho in (-0.99, 0.99)
        def nll(p):
            rho = np.tanh(p[0])            # unconstrained -> (-1,1)
            ll = np.sum(np.log(c_gaussian(u, v, rho)))
            return -ll
        res = minimize(nll, x0=[np.arctanh(0.2)], method="L-BFGS-B")
        rho = np.tanh(res.x[0])
        ll  = -res.fun
        k   = 1
        return {"name":"Gaussian", "params":{"rho":float(rho)}, "ll":float(ll), "aic":aic(ll,k), "bic":bic(ll,k,n)}

    if copula == "clayton":
        # parameter: theta > 0  -> use log transform
        def nll(p):
            th = np.exp(p[0])
            ll = np.sum(np.log(c_clayton(u, v, th)))
            return -ll
        res = minimize(nll, x0=[np.log(0.5)], method="L-BFGS-B")
        th  = np.exp(res.x[0])
        ll  = -res.fun
        k   = 1
        return {"name":"Clayton", "params":{"theta":float(th)}, "ll":float(ll), "aic":aic(ll,k), "bic":bic(ll,k,n)}

    if copula == "frank":
        # parameter: theta != 0 -> no bound, but keep away from 0 numerically
        def nll(p):
            th = p[0]
            # small |theta| can be unstable; add tiny ridge toward 0
            ll = np.sum(np.log(c_frank(u, v, th)))
            return -ll
        res = minimize(nll, x0=[2.0], method="L-BFGS-B")
        th  = res.x[0]
        ll  = -res.fun
        k   = 1
        return {"name":"Frank", "params":{"theta":float(th)}, "ll":float(ll), "aic":aic(ll,k), "bic":bic(ll,k,n)}

    raise ValueError("Unknown copula")

# ---------- 1) SEMIPARAMETRIC (empirical margins) ----------
def fit_semiparametric(df):
    data = df[['Return','LogVolume']].dropna()
    u = ecdf_u(data['Return'].values)
    v = ecdf_u(data['LogVolume'].values)

    fits = [mle_copula(u, v, "gaussian"),
            mle_copula(u, v, "clayton"),
            mle_copula(u, v, "frank")]

    # rank by AIC
    fits_sorted = sorted(fits, key=lambda d: d["aic"])
    return {"approach":"Semiparametric (empirical margins)", "n":len(u), "results":fits_sorted}

# ---------- 2) PARAMETRIC margins (IFM: fit margins, then copula on PIT) ----------
def fit_parametric(df):
    data = df[['Return','LogVolume']].dropna()
    x = data['Return'].values
    y = data['LogVolume'].values

    # Fit margins by MLE:
    # returns: Student-t (heavy tails)
    df_t, loc_t, scale_t = stats.t.fit(x)            # 3 params
    # logvolume: Normal
    mu_n, sig_n = stats.norm.fit(y)                  # 2 params

    # Transform to uniforms using fitted CDFs (Probability Integral Transform)
    u = stats.t.cdf(x, df_t, loc_t, scale_t)
    v = stats.norm.cdf(y, mu_n, sig_n)

    fits = [mle_copula(u, v, "gaussian"),
            mle_copula(u, v, "clayton"),
            mle_copula(u, v, "frank")]

    # Add margin params to k when computing model-wide AIC/BIC
    k_margins = 3 + 2   # t (df, loc, scale) + normal (mu, sigma)
    n = len(u)
    for d in fits:
        k_total = k_margins + 1     # +1 for copula parameter
        d["aic_full"] = aic(d["ll"], k_total)
        d["bic_full"] = bic(d["ll"], k_total, n)
        d["margin_params"] = {"t(df,loc,scale)": [df_t, loc_t, scale_t],
                              "norm(mu,sigma)": [mu_n, sig_n]}

    fits_sorted = sorted(fits, key=lambda d: d["aic_full"])
    return {"approach":"Parametric margins (IFM)", "n":n, "results":fits_sorted}

# ---------- 3) NONPARAMETRIC copula (empirical 2D density, pseudo log-lik) ----------
def fit_nonparametric(df, bins=25, eps=1e-9):
    data = df[['Return','LogVolume']].dropna()
    u = ecdf_u(data['Return'].values)
    v = ecdf_u(data['LogVolume'].values)

    # 2D histogram estimate on [0,1]^2 as a simple nonparametric copula density
    H, xedges, yedges = np.histogram2d(u, v, bins=bins, range=[[0,1],[0,1]], density=False)
    P = H / np.sum(H)
    # Map each (u_i, v_i) to the bin probability
    ui = np.clip((u * bins).astype(int), 0, bins-1)
    vi = np.clip((v * bins).astype(int), 0, bins-1)
    p_i = P[ui, vi] + eps
    ll = np.sum(np.log(p_i))
    # AIC/BIC are not meaningful here because #params ~ bins^2-1 grows with n; report pseudo-ll only.
    return {"approach":"Nonparametric copula (empirical 2D)", "n":len(u), "bins":bins, "pseudo_ll":float(ll)}

# ---------- RUN ALL ----------
def run_all_copulas(df):
    out1 = fit_semiparametric(df)
    out2 = fit_parametric(df)
    out3 = fit_nonparametric(df, bins=25)

    # Pretty print
    def show(block):
        print(f"\n=== {block['approach']} (n={block['n']}) ===")
        if "results" in block:
            for r in block["results"]:
                if "aic_full" in r:
                    print(f"{r['name']:9s}  ll={r['ll']:.2f}  AIC(cop)={r['aic']:.2f}  BIC(cop)={r['bic']:.2f}  "
                          f"AIC(full)={r['aic_full']:.2f}  BIC(full)={r['bic_full']:.2f}  params={r['params']}")
                else:
                    print(f"{r['name']:9s}  ll={r['ll']:.2f}  AIC={r['aic']:.2f}  BIC={r['bic']:.2f}  params={r['params']}")
        else:
            print(f"Pseudo log-likelihood = {block['pseudo_ll']:.2f}  (bins={block['bins']})")

    show(out1); show(out2); show(out3)
    return {"semiparametric": out1, "parametric": out2, "nonparametric": out3}

# ----- Example usage -----
# results = run_all_copulas(df)   # df has ['Return','LogVolume']
