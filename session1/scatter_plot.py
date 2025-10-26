import matplotlib.pyplot as plt
from scipy.stats import rankdata

def plot(returns, logvol):
    u = rankdata(returns) / (len(returns) + 1)
    v = rankdata(logvol) / (len(logvol) + 1)

    plt.figure(figsize=(6,6))
    plt.scatter(u, v, alpha=0.5, s=10, color='royalblue')
    plt.title("Scatter Plot of Integral Transformed Data (U,V)")
    plt.xlabel("U = F(Return)")
    plt.ylabel("V = F(LogVolume)")
    plt.grid(True)
    plt.savefig("result/scatter_plot.png")