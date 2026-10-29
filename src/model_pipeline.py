import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False

from src.data_generator import generate_b2b_churn_data

CATEGORICAL_FEATURES = ["service_tier"]
NUMERIC_FEATURES = [
    "contract_duration_months",
    "tenure_months",
    "monthly_retainer_idr",
    "payment_delay_days_avg",
    "unpaid_invoices_count",
    "support_tickets_count",
    "sla_breach_count",
    "scope_creep_index",
    "stakeholder_turnover",
    "qbr_attendance_rate",
    "client_satisfaction_score",
    "ad_spend_fluctuation_pct",
]
TARGET = "churned"


from src.explainer import FeatureAttributionExplainer


def get_preprocessor():
    """Build ColumnTransformer for categorical and numeric scaling."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_and_evaluate(df: pd.DataFrame, random_state: int = 42):
    """Train multiple models, benchmark, and return the best pipeline and metrics."""
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    candidate_models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=6, class_weight="balanced", random_state=random_state
        ),
    }

    if HAS_XGB:
        candidate_models["XGBoost Classifier"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=random_state,
        )
    else:
        candidate_models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=random_state,
        )

    results = {}
    best_auc = -1.0
    best_name = None
    best_pipeline = None

    for name, model in candidate_models.items():
        preprocessor = get_preprocessor()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        auc = float(roc_auc_score(y_test, y_prob))
        f1 = float(f1_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "roc_auc": round(auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "pipeline": pipeline,
        }

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipeline

    # Prepare SHAP explainer on best model
    trained_preprocessor = best_pipeline.named_steps["preprocessor"]
    trained_model = best_pipeline.named_steps["classifier"]

    # Extract transformed feature names
    cat_encoder = trained_preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_feature_names

    X_train_trans = trained_preprocessor.transform(X_train)
    X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names)

    if HAS_SHAP and shap is not None:
        if hasattr(trained_model, "get_booster") or hasattr(trained_model, "estimators_"):
            explainer = shap.TreeExplainer(trained_model)
        else:
            explainer = shap.LinearExplainer(trained_model, X_train_trans_df)
    else:
        explainer = FeatureAttributionExplainer(trained_model, feature_names)

    evaluation_summary = {
        "best_model_name": best_name,
        "best_auc": best_auc,
        "metrics_comparison": {k: {m: v[m] for m in ["roc_auc", "f1_score", "precision", "recall", "accuracy"]} for k, v in results.items()},
        "feature_names": feature_names,
    }

    return best_pipeline, explainer, evaluation_summary, (X_test, y_test)


def save_pipeline(pipeline, explainer, summary, output_dir: str = "models"):
    """Persist best pipeline, SHAP explainer, and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "explainer": explainer,
        "summary": summary,
    }
    file_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(payload, file_path)
    return file_path


def load_pipeline(model_path: str = "models/best_model.joblib"):
    """Load model artifact."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train first.")
    return joblib.load(model_path)


def get_retention_recommendations(row: dict, churn_prob: float) -> list[str]:
    """Provide prescriptive tactical retention interventions for enterprise account managers."""
    actions = []
    
    if churn_prob < 0.35:
        actions.append("Account stable. Maintain weekly cadence and schedule renewal review 60 days ahead of contract maturity.")
        return actions

    if row.get("client_satisfaction_score", 10) < 6.0:
        actions.append("Executive alignment needed: Schedule a 30-minute check-in with the client's director to address sprint concerns and reset deliverable expectations.")

    if row.get("payment_delay_days_avg", 0) > 14 or row.get("unpaid_invoices_count", 0) >= 1:
        actions.append("Payment terms review: Transition payment terms to milestone billing or offer a 3% discount for Net-7 early settlement.")

    if row.get("sla_breach_count", 0) >= 1:
        actions.append("SLA recovery: Provide a billing credit memo for the affected cycle and assign a dedicated engineer to clear the remaining defect backlog.")

    if row.get("stakeholder_turnover", 0) == 1:
        actions.append("Stakeholder onboarding: Schedule an executive onboarding workshop with the new client lead within 5 business days to walk through delivered ROI.")

    if row.get("qbr_attendance_rate", 1.0) < 0.60:
        actions.append("QBR re-engagement: Send an asynchronous 1-page executive summary covering quarterly deliverables, uptime, and ROI metrics.")

    if not actions:
        actions.append("General retention check: Conduct an informal discovery session to verify ongoing roadmap alignment and satisfaction.")

    return actions


if __name__ == "__main__":
    data_file = "data/synthetic_b2b_churn.csv"
    if not os.path.exists(data_file):
        print("Generating dataset...")
        df = generate_b2b_churn_data(1000)
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)

    print(f"Dataset ready with {len(df)} records. Training models...")
    pipeline, explainer, summary, _ = train_and_evaluate(df)
    print("\n--- Model Benchmark Results ---")
    for name, metrics in summary["metrics_comparison"].items():
        print(f"[{name}] AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f} | Recall: {metrics['recall']:.4f}")

    print(f"\nWinner: {summary['best_model_name']} with AUC {summary['best_auc']:.4f}")
    saved_path = save_pipeline(pipeline, explainer, summary)
    print(f"Artifacts successfully saved to {saved_path}")
