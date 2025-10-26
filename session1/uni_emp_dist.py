import matplotlib.pyplot as plt
import seaborn as sns


def plot(returns, logvol):
    plt.figure(figsize=(12, 5))

    # Plot 1: Return distribution
    plt.subplot(1, 2, 1)
    sns.histplot(returns, bins=60, kde=True, color='teal')
    plt.title("Empirical Distribution of Tesla Daily Returns")
    plt.xlabel("Return")
    plt.ylabel("Density")

    # Plot 2: LogVolume distribution
    plt.subplot(1, 2, 2)
    sns.histplot(logvol, bins=60, kde=True, color='teal')
    plt.title("Empirical Distribution of Tesla Log Trading Volume")
    plt.xlabel("Log(Volume)")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig("result/hist_plot.png")