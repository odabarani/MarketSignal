import yfinance as yf
import pandas as pd
from datetime import date

START = "2015-01-01"

def _strip_tz(df):
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def get_stock_data(ticker):
    df  = yf.download(ticker, start=START, end=date.today(), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = _strip_tz(df)

    vix = yf.download("^VIX", start=START, end=date.today(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = _strip_tz(vix)
    df['VIX'] = vix['Close']

    spy = yf.download("SPY", start=START, end=date.today(), progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = _strip_tz(spy)
    df['SPY_Close'] = spy['Close']

    try:
        info = yf.Ticker(ticker).info
        df['PE_Ratio']       = info.get('trailingPE', 0)
        df['Profit_Margin']  = info.get('profitMargins', 0)
        df['Revenue_Growth'] = info.get('revenueGrowth', 0)
        df['Debt_Equity']    = info.get('debtToEquity', 0)
        for col in ['PE_Ratio', 'Profit_Margin', 'Revenue_Growth', 'Debt_Equity']:
            df[col] = df[col].ffill()
    except Exception:
        for col in ['PE_Ratio', 'Profit_Margin', 'Revenue_Growth', 'Debt_Equity']:
            df[col] = 0

    df.dropna(subset=['VIX', 'SPY_Close'], inplace=True)
    return df
