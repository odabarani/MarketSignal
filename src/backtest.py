import numpy as np

def backtest(df, model, features):
    cash             = 10000
    position         = 0
    portfolio_values = []
    split            = int(len(df) * 0.7)  # only test on unseen 30%

    for i in range(split, len(df) - 1):
        row           = df.iloc[i]
        X             = row[features].values.reshape(1, -1)
        prediction    = model.predict(X)[0]
        confidence    = max(model.predict_proba(X)[0])
        current_price = float(row['Close'].item())

        if prediction == 1 and confidence > 0.65 and position == 0:
            position = cash / current_price
            cash     = 0
        elif prediction == 0 and confidence > 0.65 and position > 0:
            cash     = position * current_price
            position = 0

        value = cash + position * current_price
        portfolio_values.append(value)

    return portfolio_values

def calculate_metrics(portfolio_values):
    values  = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    peak         = np.maximum.accumulate(values)
    drawdown     = (values - peak) / peak
    max_drawdown = drawdown.min() * 100

    return round(sharpe, 2), round(max_drawdown, 1)
