import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

def create_model(data: str):
    df = pd.read_csv(data)
    y = df['HousingPrice']
    X = df[['InterestRate', 'Income', 'Unemployment']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def report(model):
    print(model.summary())

    # display key stats neatly
    print(f"\nR² = {model.rsquared:.4f}, Adjusted R² = {model.rsquared_adj:.4f}")
    print(f"AIC = {model.aic:.2f}, BIC = {model.bic:.2f}")

def residual_graph(model):
    fitted = model.fittedvalues
    residuals = model.resid

    plt.figure(figsize=(7,5))
    plt.scatter(fitted, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.savefig('residual_graph.png')

def qq_plot(model):
    import statsmodels.api as sm
    residuals = model.resid
    sm.qqplot(residuals, line='45', fit=True)
    plt.title('QQ Plot of Residuals')
    plt.savefig('qq_plot.png')

def hist_plot(model):
    residuals = model.resid
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals')
    plt.savefig('hist_plot.png')

def time_series_plot(model):
    residuals = model.resid
    plt.plot(residuals)
    plt.title('Residuals over Time')
    plt.xlabel('Observation Index')
    plt.ylabel('Residual')
    plt.savefig('time_series_plot.png')


def VIF(data: str):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    df = pd.read_csv(data)
    X_no_const = df[['InterestRate', 'Income', 'Unemployment']]
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X_no_const.columns
    vif_data['VIF'] = [variance_inflation_factor(X_no_const.values, i) for i in range(X_no_const.shape[1])]
    print(vif_data)


if __name__ == '__main__':
    data_path = '../data_session2/merged_housing_macro.csv'
    # model = create_model(data_path)

    # report(model)

    # residual_graph(model)

    # qq_plot(model)

    # hist_plot(model)

    # time_series_plot(model)

    VIF(data_path)

