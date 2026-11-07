import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

DATA_PATH = "data/processed/diabetes_parquet.parquet"

def main():
    df = pd.read_parquet(DATA_PATH)
    print("Dataset loaded:", df.shape)

    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        start = time.time()
        if name == "Gradient Boosting":
            weights = compute_sample_weight(
                class_weight="balanced", y=y_train
            )
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model.fit(X_train, y_train)

        elapsed = time.time() - start
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1_Score": f1_score(y_test, prediction, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probability),
            "Training_Time_Seconds": elapsed
        })
        trained_models[name] = model
        print(name, results[-1])

    results_df = pd.DataFrame(results)
    best_name = results_df.loc[results_df["F1_Score"].idxmax(), "Model"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(trained_models[best_name], "models/best_model.joblib")
    results_df.to_csv("results/model_comparison.csv", index=False)

    print("Best model based on F1-score:", best_name)
    print("Saved model: models/best_model.joblib")
    print("Saved metrics: results/model_comparison.csv")

if __name__ == "__main__":
    main()
