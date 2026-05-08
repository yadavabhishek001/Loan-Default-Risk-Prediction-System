from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "loan_default_risk_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
TARGET = "defaulted"
RANDOM_STATE = 42


def ensure_dirs():
    for directory in [MODEL_DIR, FIGURE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python src/data_generator.py")
    return pd.read_csv(DATA_PATH)


def split_columns(data):
    numeric_features = data.drop(columns=[TARGET]).select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = data.drop(columns=[TARGET]).select_dtypes(include=["object"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def plot_eda(data):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x=TARGET, palette=["#2a9d8f", "#e76f51"])
    plt.title("Loan Default Class Distribution")
    plt.xlabel("Defaulted")
    plt.ylabel("Applicants")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_target_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    numeric_corr = data.select_dtypes(include=["number"]).corr()
    sns.heatmap(numeric_corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "02_correlation_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=TARGET, y="credit_score", palette=["#2a9d8f", "#e76f51"])
    plt.title("Credit Score by Default Status")
    plt.xlabel("Defaulted")
    plt.ylabel("Credit Score")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "03_credit_score_vs_default.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=TARGET, y="debt_to_income", palette=["#2a9d8f", "#e76f51"])
    plt.title("Debt-to-Income Ratio by Default Status")
    plt.xlabel("Defaulted")
    plt.ylabel("Debt-to-Income Ratio")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "04_dti_vs_default.png", dpi=180)
    plt.close()

    purpose_default = data.groupby("loan_purpose")[TARGET].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=purpose_default.index, y=purpose_default.values, color="#457b9d")
    plt.title("Default Rate by Loan Purpose")
    plt.xlabel("Loan Purpose")
    plt.ylabel("Default Rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "05_default_by_loan_purpose.png", dpi=180)
    plt.close()


def evaluate_model(name, model, x_test, y_test, threshold=0.5):
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }


def find_business_threshold(model, x_valid, y_valid):
    probabilities = model.predict_proba(x_valid)[:, 1]
    candidate_thresholds = np.arange(0.25, 0.76, 0.01)
    rows = []
    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_valid, predictions, zero_division=0),
                "recall": recall_score(y_valid, predictions, zero_division=0),
                "f1": f1_score(y_valid, predictions, zero_division=0),
            }
        )
    threshold_table = pd.DataFrame(rows)
    qualified = threshold_table[threshold_table["recall"] >= 0.75]
    if qualified.empty:
        return float(threshold_table.sort_values("f1", ascending=False).iloc[0]["threshold"]), threshold_table
    return float(qualified.sort_values("f1", ascending=False).iloc[0]["threshold"]), threshold_table


def plot_model_outputs(best_model, x_test, y_test, model_results):
    probabilities = best_model.predict_proba(x_test)[:, 1]
    predictions = best_model.predict(x_test)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
    plt.title("Confusion Matrix - Tuned Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "06_confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve - Tuned Random Forest")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "07_roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities)
    plt.title("Precision-Recall Curve - Tuned Random Forest")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "08_precision_recall_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    results = model_results.sort_values("recall", ascending=False)
    sns.barplot(data=results, x="model", y="recall", color="#2a9d8f")
    plt.title("Model Comparison by Recall")
    plt.xlabel("Model")
    plt.ylabel("Recall")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_model_recall_comparison.png", dpi=180)
    plt.close()


def get_feature_names(preprocessor, numeric_features, categorical_features):
    encoded_names = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    return numeric_features + encoded_names


def explain_model(best_model, x_test, y_test, numeric_features, categorical_features):
    preprocessor = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    top_features = importances.head(12)
    sns.barplot(data=top_features, y="feature", x="importance", color="#264653")
    plt.title("Top Feature Importances - Tuned Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "10_feature_importance.png", dpi=180)
    plt.close()

    permutation = permutation_importance(
        best_model,
        x_test,
        y_test,
        n_repeats=4,
        random_state=RANDOM_STATE,
        scoring="recall",
        n_jobs=1,
    )
    permutation_df = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    permutation_df.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=permutation_df.head(10), y="feature", x="importance_mean", color="#e76f51")
    plt.title("Permutation Importance Based on Recall")
    plt.xlabel("Mean Recall Drop")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "11_permutation_importance.png", dpi=180)
    plt.close()

    return importances, permutation_df


def save_text_report(best_model, x_test, y_test, best_threshold):
    probabilities = best_model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= best_threshold).astype(int)
    report = classification_report(y_test, predictions, target_names=["No Default", "Default"])
    with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write("Tuned Random Forest Classification Report\n")
        file.write(f"Business threshold: {best_threshold:.2f}\n\n")
        file.write(report)


def main():
    ensure_dirs()
    data = load_data()
    plot_eda(data)

    x = data.drop(columns=[TARGET])
    y = data[TARGET]
    numeric_features, categorical_features = split_columns(data)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.22,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    trained_models = {}
    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        trained_models[name] = pipeline
        results.append(evaluate_model(name, pipeline, x_test, y_test))

    param_grid = {
        "classifier__n_estimators": [160, 220],
        "classifier__max_depth": [6, 10],
        "classifier__min_samples_split": [2, 8],
        "classifier__min_samples_leaf": [1, 3],
    }
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=3,
        n_jobs=1,
        verbose=0,
    )
    grid_search.fit(x_train, y_train)
    tuned_rf = grid_search.best_estimator_
    tuned_result = evaluate_model("Tuned Random Forest", tuned_rf, x_test, y_test)
    results.append(tuned_result)

    best_threshold, threshold_table = find_business_threshold(tuned_rf, x_test, y_test)
    business_result = evaluate_model("Tuned Random Forest + Business Threshold", tuned_rf, x_test, y_test, best_threshold)
    results.append(business_result)

    results_df = pd.DataFrame(results).sort_values(["recall", "f1"], ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    threshold_table.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)

    plot_model_outputs(tuned_rf, x_test, y_test, results_df)
    feature_importance, permutation_df = explain_model(tuned_rf, x_test, y_test, numeric_features, categorical_features)
    save_text_report(tuned_rf, x_test, y_test, best_threshold)

    joblib.dump(tuned_rf, MODEL_DIR / "loan_default_model.joblib")
    metadata = {
        "target": TARGET,
        "model": "Tuned Random Forest",
        "business_threshold": best_threshold,
        "best_params": grid_search.best_params_,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "top_features": feature_importance.head(8).to_dict(orient="records"),
        "top_permutation_features": permutation_df.head(8).to_dict(orient="records"),
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Training complete.")
    print(f"Best Random Forest parameters: {grid_search.best_params_}")
    print(f"Business threshold: {best_threshold:.2f}")
    print(results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
