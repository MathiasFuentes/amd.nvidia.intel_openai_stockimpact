import yfinance as yf

yf.download("AMD", start="2020-01-01", end="2025-08-18", interval="1d",
            auto_adjust=True, progress=False).to_csv("AMD_stock_2020_2025.csv")

yf.download("INTC", start="2020-01-01", end="2025-08-18", interval="1d",
            auto_adjust=True, progress=False).to_csv("INTC_stock_2020_2025.csv")
