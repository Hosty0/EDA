import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# 1. Load trained model
model = joblib.load("/artifacts/model.joblib")

# 2. Load dataset (как в архиве)
abalone = fetch_ucirepo(id=1)
X = abalone.data.features.copy()
y = abalone.data.targets.iloc[:, 0]

# 3. Preprocessing (совпадает с train)
X = pd.get_dummies(X, columns=["Sex"], drop_first=False)

# 4. Create explainer
explainer = RegressionExplainer(
    model,
    X,
    y
)

# 5. 🔑 SIMPLE dashboard (БЕЗ what-if, БЕЗ input)
dashboard = ExplainerDashboard(
    explainer,
    title="Abalone Age Prediction (RandomForest)",
    model_summary="RandomForestRegressor",
    simple=True
)

# 6. Run dashboard
dashboard.run(
    host="0.0.0.0",
    port=8050,
    use_waitress=True
)