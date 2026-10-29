import os
import numpy as np
import pandas as pd
import pytest
from src.data_generator import generate_b2b_churn_data
from src.model_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_retention_recommendations,
    save_pipeline,
    train_and_evaluate,
)


def test_data_generator_shape_and_distribution():
    """Verify data generator generates valid, complete, and balanced synthetic data."""
    df = generate_b2b_churn_data(n_samples=250, random_state=123)
    
    assert len(df) == 250
    assert not df.isnull().any().any()
    
    expected_cols = ["client_id"] + CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["churned"]
    for col in expected_cols:
        assert col in df.columns

    churn_rate = df["churned"].mean()
    assert 0.10 <= churn_rate <= 0.40, f"Churn rate {churn_rate:.2f} is out of realistic expected bounds."


def test_model_training_and_shap_evaluation(tmp_path):
    """Verify model pipeline trains models, computes metrics, and produces working SHAP explainer."""
    df = generate_b2b_churn_data(n_samples=300, random_state=42)
    pipeline, explainer, summary, (X_test, y_test) = train_and_evaluate(df, random_state=42)

    assert summary["best_auc"] >= 0.75, f"Best model AUC was {summary['best_auc']}, expected >= 0.75"
    assert "best_model_name" in summary
    assert len(summary["feature_names"]) > len(NUMERIC_FEATURES)

    # Test single prediction
    sample_client = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].iloc[[0]]
    prob = pipeline.predict_proba(sample_client)[0, 1]
    assert 0.0 <= prob <= 1.0

    # Test SHAP explanation shape
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_sample = preprocessor.transform(sample_client)
    shap_vals = explainer.shap_values(transformed_sample)
    
    if isinstance(shap_vals, list):
        row_shap = shap_vals[1][0]
    else:
        row_shap = shap_vals[0]
    
    assert len(row_shap) == len(summary["feature_names"])

    # Test saving
    save_path = save_pipeline(pipeline, explainer, summary, output_dir=str(tmp_path))
    assert os.path.exists(save_path)


def test_retention_recommendations():
    """Verify recommendation engine produces actionable prescriptive guidance."""
    healthy_client = {
        "client_satisfaction_score": 9.5,
        "payment_delay_days_avg": 2.0,
        "sla_breach_count": 0,
        "stakeholder_turnover": 0,
        "qbr_attendance_rate": 0.95,
    }
    recs_healthy = get_retention_recommendations(healthy_client, churn_prob=0.15)
    assert any("stable" in r.lower() for r in recs_healthy)

    at_risk_client = {
        "client_satisfaction_score": 3.0,
        "payment_delay_days_avg": 35.0,
        "sla_breach_count": 3,
        "stakeholder_turnover": 1,
        "qbr_attendance_rate": 0.20,
    }
    recs_risk = get_retention_recommendations(at_risk_client, churn_prob=0.88)
    assert len(recs_risk) >= 3
    assert any("alignment" in r.lower() for r in recs_risk)
    assert any("payment" in r.lower() for r in recs_risk)
    assert any("sla" in r.lower() for r in recs_risk)
