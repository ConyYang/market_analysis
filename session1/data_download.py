import yfinance as yf

# # S&P 500 index
# gspc = yf.download("^GSPC", start="2015-10-20", end="2025-10-19")
# gspc.to_csv("SP500_data.csv")

# Example stock: Apple
# aapl = yf.download("AAPL", start="2015-10-20", end="2025-10-19")
# aapl.to_csv("AAPL_data.csv")

# Example stock: Microsoft
msft = yf.download("MSFT", start="2015-10-23", end="2025-10-22")
msft.to_csv("MSFT_data.csv")

# Example stock: Tesla
msft = yf.download("TSLA", start="2015-10-23", end="2025-10-22")
msft.to_csv("TSLA_data.csv")