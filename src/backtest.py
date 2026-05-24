import numpy as np

def backtest(df, model, features):
    cash             = 10000
    position         = 0
    portfolio_values = []
    split            = int(len(df) * 0.7)
    df_test          = df.iloc[split:]

    # batch predict everything at once instead of row by row
    X_all         = df_test[features].values
    predictions   = model.predict(X_all)
    probabilities = model.predict_proba(X_all)
    confidences   = probabilities.max(axis=1)

    for i in range(len(df_test) - 1):
        current_price = float(df_test['Close'].iloc[i].item())
        prediction    = predictions[i]
        confidence    = confidences[i]

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
