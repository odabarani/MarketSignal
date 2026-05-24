from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_model(df, features):
    X = df[features]
    y = df['Target']

    split   = int(len(X) * 0.70)
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test  = X.iloc[split:]
    y_test  = y.iloc[split:]

    if len(X_test) == 0:
        raise ValueError("Not enough historical data to test this ticker. Try a major US stock like AAPL or NVDA.")
    if y_test.nunique() < 2:
        raise ValueError("Test set contains only one price direction. Try a different ticker or date range.")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return model, accuracy
