from session1.copula.helper_function import to_u_v
from session1.copula.parametric_frank import frank_copula_mle
from session1.copula.parametric_gaussian import *
from session1.copula.parametric_student_t import t_copula_mle
from session1.process_file import standardize_data

# Task 5
if __name__ == '__main__':
    processed_df = standardize_data("../../data/TSLA_data.csv")
    df_clean = processed_df[['Return', 'LogVolume']].dropna()
    returns = df_clean['Return'].to_numpy()
    volume  = df_clean['LogVolume'].to_numpy()
    length = len(returns)
    x, y = to_u_v(returns, volume)

    # Gaussian Copula
    res_g = gaussian_copula_mle(x, y, length)
    for k, v in res_g.items():
        print(f"{k}: {float(v)}")


    # Frank Copula
    res_f = frank_copula_mle(x, y, length)
    for k, v in res_f.items():
        print(f"{k}: {float(v)}")

    #Student t Copula
    res_f = t_copula_mle(x, y, length)
    for k, v in res_f.items():
        print(f"{k}: {v}")
