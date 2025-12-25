import json
import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ART_DIR = "/artifacts"

def prepare_training_data(df):
    y = df["Rings"]
    X = df.drop(columns=["Rings"], errors="ignore")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    sex_ohe = encoder.fit_transform(X[["Sex"]])
    sex_cols = encoder.get_feature_names_out(["Sex"])

    X = pd.concat(
        [
            X.drop(columns=["Sex"]).reset_index(drop=True),
            pd.DataFrame(sex_ohe, columns=sex_cols)
        ],
        axis=1
    )

    with open(os.path.join(ART_DIR, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    with open(os.path.join(ART_DIR, "feature_cols.json"), "w") as f:
        json.dump(list(X.columns), f)

    return X, y