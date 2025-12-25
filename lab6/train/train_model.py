import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from ucimlrepo import fetch_ucirepo

from eda import prepare_training_data

ART_DIR = "/artifacts"
os.makedirs(ART_DIR, exist_ok=True)

abalone = fetch_ucirepo(id=1)
df = abalone.data.features
df["Rings"] = abalone.data.targets.iloc[:, 0]

X, y = prepare_training_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.85, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

joblib.dump(model, os.path.join(ART_DIR, "model.joblib"))
print("Model trained and saved.")