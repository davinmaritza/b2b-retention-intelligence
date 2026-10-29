import numpy as np
import pandas as pd

def generate_b2b_churn_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic B2B retainer client data for enterprise portfolio analysis."""
    np.random.seed(random_state)
    
    services = [
        "Custom Enterprise Software",
        "Dedicated AI & Data Squad",
        "Performance Marketing & Digital Retainer",
    ]
    
    # 1. Firmographics & Contract features
    service_tier = np.random.choice(services, size=n_samples, p=[0.45, 0.25, 0.30])
    contract_duration_months = np.random.choice([3, 6, 12, 24], size=n_samples, p=[0.15, 0.30, 0.45, 0.10])
    tenure_months = np.random.randint(1, 37, size=n_samples)
    
    # Retainer fee depends slightly on service tier
    base_mrr = np.where(
        service_tier == "Dedicated AI & Data Squad",
        np.random.normal(85.0, 20.0, size=n_samples),
        np.where(
            service_tier == "Custom Enterprise Software",
            np.random.normal(60.0, 15.0, size=n_samples),
            np.random.normal(35.0, 10.0, size=n_samples)
        )
    )
    monthly_retainer_idr = np.clip(np.round(base_mrr, 1), 15.0, 250.0)
    
    # 2. Operational & Billing Friction Signals
    payment_delay_days_avg = np.clip(np.random.exponential(scale=7.0, size=n_samples), 0, 60).round(1)
    unpaid_invoices_count = np.random.poisson(lam=0.4, size=n_samples)
    support_tickets_count = np.random.poisson(lam=8.0, size=n_samples)
    sla_breach_count = np.random.poisson(lam=0.8, size=n_samples)
    scope_creep_index = np.clip(np.random.normal(2.5, 0.9, size=n_samples), 1.0, 5.0).round(1)
    
    # 3. Client Relationship & Engagement
    stakeholder_turnover = np.random.binomial(n=1, p=0.22, size=n_samples)
    qbr_attendance_rate = np.clip(np.random.beta(a=5, b=2, size=n_samples), 0.0, 1.0).round(2)
    client_satisfaction_score = np.clip(np.random.normal(7.2, 1.6, size=n_samples), 1.0, 10.0).round(1)
    ad_spend_fluctuation_pct = np.clip(np.random.normal(5.0, 20.0, size=n_samples), -50.0, 50.0).round(1)
    
    # 4. Latent Churn Propensity (Realistic Logistic Function)
    # Higher risk when: CSAT is low, payment delayed, SLA breached, PIC left, QBR missed
    z = (
        -1.8
        - 0.55 * (client_satisfaction_score - 7.0)
        + 0.06 * payment_delay_days_avg
        + 0.45 * sla_breach_count
        + 0.60 * unpaid_invoices_count
        + 0.75 * stakeholder_turnover
        - 1.20 * (qbr_attendance_rate - 0.7)
        - 0.03 * (tenure_months - 12)
        - 0.015 * ad_spend_fluctuation_pct
        + 0.15 * (scope_creep_index - 2.5)
        + np.random.normal(0, 0.4, size=n_samples)
    )
    
    prob_churn = 1 / (1 + np.exp(-z))
    churned = (prob_churn >= 0.50).astype(int)
    
    client_ids = [f"ACC-{1000 + i}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        "client_id": client_ids,
        "service_tier": service_tier,
        "contract_duration_months": contract_duration_months,
        "tenure_months": tenure_months,
        "monthly_retainer_idr": monthly_retainer_idr,
        "payment_delay_days_avg": payment_delay_days_avg,
        "unpaid_invoices_count": unpaid_invoices_count,
        "support_tickets_count": support_tickets_count,
        "sla_breach_count": sla_breach_count,
        "scope_creep_index": scope_creep_index,
        "stakeholder_turnover": stakeholder_turnover,
        "qbr_attendance_rate": qbr_attendance_rate,
        "client_satisfaction_score": client_satisfaction_score,
        "ad_spend_fluctuation_pct": ad_spend_fluctuation_pct,
        "churned": churned,
    })
    
    return df

if __name__ == "__main__":
    df = generate_b2b_churn_data(1000)
    print(f"Generated {len(df)} records. Churn rate: {df['churned'].mean():.2%}")
    df.to_csv("data/synthetic_b2b_churn.csv", index=False)
    print("Saved to data/synthetic_b2b_churn.csv")
