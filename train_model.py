"""Train the vehicle recommendation model.
Run this locally, or copy the code into a Kaggle notebook and run there.
It saves model/vehicle_model.joblib and model/vehicle_data.csv.
"""
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import joblib

DATA_PATH = "data/vehicles.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "vehicle_model.joblib")
DATA_OUT_PATH = os.path.join(MODEL_DIR, "vehicle_data.csv")


def build_model(df: pd.DataFrame):
    # Features the model uses to match a user's budget & preferences
    feature_cols = ["Price_USD", "Mileage_kmpl", "Seats", "Advantage_Score"]
    categorical_cols = ["Type", "Fuel_Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), feature_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    nn = NearestNeighbors(n_neighbors=min(5, len(df)), metric="euclidean")

    pipe = Pipeline(steps=[("prep", preprocessor), ("nn", nn)])
    pipe.fit(df[feature_cols + categorical_cols])
    return pipe, feature_cols, categorical_cols


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    pipe, feature_cols, categorical_cols = build_model(df)

    joblib.dump(
        {
            "pipe": pipe,
            "feature_cols": feature_cols,
            "categorical_cols": categorical_cols,
        },
        MODEL_PATH,
    )
    df.to_csv(DATA_OUT_PATH, index=False)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Data saved to {DATA_OUT_PATH}")


if __name__ == "__main__":
    main()
