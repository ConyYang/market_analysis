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

if __name__ == '__main__':
    model = create_model("../data_session2/merged_housing_macro.csv")

    # report(model)
    residual_graph(model)


