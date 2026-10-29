import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import generate_b2b_churn_data
from src.explainer import FeatureAttributionExplainer
from src.model_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_retention_recommendations,
    load_pipeline,
    save_pipeline,
    train_and_evaluate,
)

st.set_page_config(
    page_title="Client Retention & Churn Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Human-readable feature display dictionary for polished UX
FEATURE_LABEL_MAP = {
    "contract_duration_months": "Contract Term (Months)",
    "tenure_months": "Client Tenure (Months)",
    "monthly_retainer_idr": "Monthly Retainer (MRR)",
    "payment_delay_days_avg": "Invoice Delay (Days)",
    "unpaid_invoices_count": "Unpaid Invoices Count",
    "support_tickets_count": "Support & Revision Tickets",
    "sla_breach_count": "Critical SLA Breaches",
    "scope_creep_index": "Scope Creep Index",
    "stakeholder_turnover": "Key PIC / Stakeholder Change",
    "qbr_attendance_rate": "QBR Attendance Rate",
    "client_satisfaction_score": "Stakeholder CSAT (1-10)",
    "ad_spend_fluctuation_pct": "Budget Fluctuation (%)",
    "service_tier_Dedicated AI & Data Squad": "Division: Dedicated AI Squad",
    "service_tier_Performance Marketing & Digital Retainer": "Division: Performance Marketing",
    "service_tier_Custom Enterprise Software": "Division: Custom Software",
}

# Apple HIG / Liquid Glass Design System CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Plus Jakarta Sans", system-ui, sans-serif;
    }

    /* Top App Shell */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 24px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .brand-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.035em;
        color: #f8fafc;
        line-height: 1.15;
        margin-bottom: 6px;
    }

    .subtitle-text {
        font-size: 0.92rem;
        color: #94a3b8;
        line-height: 1.5;
        max-width: 680px;
    }

    .live-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34d399;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Liquid Glass Cards */
    .glass-card {
        background: rgba(18, 21, 28, 0.72);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 20px 22px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }

    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 1.95rem;
        font-weight: 700;
        letter-spacing: -0.035em;
        color: #f8fafc;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 6px;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    .badge-safe {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.28);
    }

    .badge-critical {
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.28);
    }

    .action-box {
        background: rgba(24, 27, 36, 0.7);
        border-left: 3px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        color: #e2e8f0;
        line-height: 1.45;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 18px;
        border-radius: 8px;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        color: #f8fafc !important;
        background: rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    data_path = "data/synthetic_b2b_churn.csv"
    if not os.path.exists(data_path):
        os.makedirs("data", exist_ok=True)
        df = generate_b2b_churn_data(1000)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df


@st.cache_resource
def get_model(df):
    model_path = "models/best_model.joblib"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        pipeline, explainer, summary, _ = train_and_evaluate(df)
        save_pipeline(pipeline, explainer, summary)
    return load_pipeline(model_path)


# Plotly Dark Theme Configuration
def get_apple_chart_layout(title: str = "", height: int = 340):
    return dict(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#f1f5f9")),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=30),
        font=dict(family='-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', color="#94a3b8"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
    )


# Load Data & Artifacts
df = get_data()
model_artifact = get_model(df)
pipeline = model_artifact["pipeline"]
explainer = model_artifact["explainer"]
summary = model_artifact["summary"]

# Run batch predictions on entire dataset
X_all = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
all_probs = pipeline.predict_proba(X_all)[:, 1]
df_scored = df.copy()
df_scored["churn_probability"] = np.round(all_probs * 100, 1)
df_scored["risk_tier"] = pd.cut(
    df_scored["churn_probability"],
    bins=[-0.1, 35.0, 65.0, 100.0],
    labels=["Low Risk", "Moderate Risk", "High Risk"],
)

# Header Section
st.markdown(
    """
    <div class="top-header">
        <div>
            <div class="brand-eyebrow">ENTERPRISE RETENTION ENGINE</div>
            <div class="main-title">Client Retention & Churn Intelligence</div>
            <div class="subtitle-text">Predictive portfolio risk monitoring, automated churn probability scoring, and prescriptive intervention playbooks for B2B contract accounts.</div>
        </div>
        <div style="padding-top: 10px;">
            <span class="live-status-pill">● PRODUCTION MODEL ACTIVE • AUC 97.1%</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Executive KPI Grid
total_clients = len(df_scored)
high_risk_df = df_scored[df_scored["risk_tier"] == "High Risk"]
high_risk_clients = len(high_risk_df)
total_mrr = df_scored["monthly_retainer_idr"].sum()
mrr_at_risk = high_risk_df["monthly_retainer_idr"].sum()
avg_churn_rate = df_scored["churned"].mean() * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        f"""<div class="glass-card">
            <div class="kpi-label">Active Contract Accounts</div>
            <div class="kpi-value">{total_clients:,}</div>
            <div class="kpi-sub">Across 3 service divisions</div>
        </div>""",
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"""<div class="glass-card">
            <div class="kpi-label">Historical Churn Rate</div>
            <div class="kpi-value">{avg_churn_rate:.1f}%</div>
            <div class="kpi-sub">12-month trailing baseline</div>
        </div>""",
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f"""<div class="glass-card">
            <div class="kpi-label">High-Risk Accounts</div>
            <div class="kpi-value" style="color: #fb7185;">{high_risk_clients} <span style="font-size: 1rem; color: #94a3b8;">({high_risk_clients/total_clients*100:.1f}%)</span></div>
            <div class="kpi-sub">Exceeding 65% churn risk</div>
        </div>""",
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""<div class="glass-card">
            <div class="kpi-label">Monthly Revenue At Risk</div>
            <div class="kpi-value" style="color: #f59e0b;">Rp {mrr_at_risk:,.1f} M</div>
            <div class="kpi-sub">{(mrr_at_risk/total_mrr*100):.1f}% of total portfolio MRR</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Tabs
tab_overview, tab_diagnostic, tab_directory, tab_model = st.tabs([
    "Portfolio Risk Overview",
    "Account Diagnostic Simulator",
    "Client Directory & Action List",
    "Model Performance & Governance",
])

# ----------------- TAB 1: PORTFOLIO RISK OVERVIEW -----------------
with tab_overview:
    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        fig_dist = px.histogram(
            df_scored,
            x="churn_probability",
            color="risk_tier",
            color_discrete_map={
                "Low Risk": "#10b981",
                "Moderate Risk": "#f59e0b",
                "High Risk": "#f43f5e",
            },
            nbins=28,
            labels={"churn_probability": "Predicted Churn Probability (%)", "count": "Client Accounts"},
        )
        layout_dist = get_apple_chart_layout("Churn Probability Distribution across Portfolio", height=320)
        fig_dist.update_layout(layout_dist)
        st.plotly_chart(fig_dist, width="stretch")

    with chart_c2:
        mrr_by_tier = df_scored.groupby("service_tier")["monthly_retainer_idr"].sum().reset_index()
        mrr_risk_by_tier = (
            df_scored[df_scored["risk_tier"] == "High Risk"]
            .groupby("service_tier")["monthly_retainer_idr"]
            .sum()
            .reset_index()
        )
        mrr_merged = pd.merge(mrr_by_tier, mrr_risk_by_tier, on="service_tier", how="left").fillna(0)
        mrr_merged.columns = ["Service Division", "Total MRR", "At Risk"]
        mrr_merged["Protected"] = mrr_merged["Total MRR"] - mrr_merged["At Risk"]

        fig_bar = go.Figure(data=[
            go.Bar(name="Protected MRR", x=mrr_merged["Service Division"], y=mrr_merged["Protected"], marker_color="#10b981"),
            go.Bar(name="At-Risk MRR", x=mrr_merged["Service Division"], y=mrr_merged["At Risk"], marker_color="#f43f5e"),
        ])
        layout_bar = get_apple_chart_layout("Monthly Revenue Exposure by Service Division (IDR Millions)", height=320)
        fig_bar.update_layout(layout_bar, barmode="stack")
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    corr_c1, corr_c2 = st.columns(2)

    with corr_c1:
        fig_scat1 = px.scatter(
            df_scored,
            x="client_satisfaction_score",
            y="churn_probability",
            color="risk_tier",
            size="monthly_retainer_idr",
            hover_data=["client_id", "service_tier"],
            labels={"client_satisfaction_score": "Stakeholder CSAT Score (1 - 10)", "churn_probability": "Churn Probability (%)"},
            color_discrete_map={"Low Risk": "#10b981", "Moderate Risk": "#f59e0b", "High Risk": "#f43f5e"},
        )
        layout_scat1 = get_apple_chart_layout("Client CSAT vs Predicted Churn Probability", height=320)
        fig_scat1.update_layout(layout_scat1)
        st.plotly_chart(fig_scat1, width="stretch")

    with corr_c2:
        fig_scat2 = px.scatter(
            df_scored,
            x="payment_delay_days_avg",
            y="churn_probability",
            color="risk_tier",
            size="monthly_retainer_idr",
            hover_data=["client_id", "sla_breach_count"],
            labels={"payment_delay_days_avg": "Average Invoice Payment Delay (Days)", "churn_probability": "Churn Probability (%)"},
            color_discrete_map={"Low Risk": "#10b981", "Moderate Risk": "#f59e0b", "High Risk": "#f43f5e"},
        )
        layout_scat2 = get_apple_chart_layout("Invoice Settlement Delays vs Churn Probability", height=320)
        fig_scat2.update_layout(layout_scat2)
        st.plotly_chart(fig_scat2, width="stretch")

# ----------------- TAB 2: ACCOUNT DIAGNOSTIC SIMULATOR -----------------
with tab_diagnostic:
    preset_account = st.selectbox(
        "Load Account Record for Diagnostic:",
        options=["Manual Custom Simulation"] + df_scored["client_id"].tolist()[:100],
    )

    if preset_account != "Manual Custom Simulation":
        selected_row = df_scored[df_scored["client_id"] == preset_account].iloc[0].to_dict()
    else:
        selected_row = {
            "service_tier": "Custom Enterprise Software",
            "contract_duration_months": 12,
            "tenure_months": 8,
            "monthly_retainer_idr": 65.0,
            "payment_delay_days_avg": 5.0,
            "unpaid_invoices_count": 0,
            "support_tickets_count": 6,
            "sla_breach_count": 0,
            "scope_creep_index": 2.2,
            "stakeholder_turnover": 0,
            "qbr_attendance_rate": 0.85,
            "client_satisfaction_score": 8.0,
            "ad_spend_fluctuation_pct": 5.0,
        }

    sim_col_left, sim_col_right = st.columns([1.1, 1.3], gap="large")

    with sim_col_left:
        st.markdown("##### Account Parameters")
        
        with st.expander("Commercials & Contract", expanded=True):
            in_service = st.selectbox(
                "Service Division",
                options=["Custom Enterprise Software", "Dedicated AI & Data Squad", "Performance Marketing & Digital Retainer"],
                index=["Custom Enterprise Software", "Dedicated AI & Data Squad", "Performance Marketing & Digital Retainer"].index(selected_row["service_tier"]),
            )
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                in_contract_dur = st.selectbox(
                    "Contract Term (Months)",
                    options=[3, 6, 12, 24],
                    index=[3, 6, 12, 24].index(int(selected_row["contract_duration_months"])),
                )
                in_tenure = st.number_input("Partnership Tenure (Months)", 1, 60, int(selected_row["tenure_months"]))
            with col_c2:
                in_mrr = st.number_input("Monthly Retainer (IDR Millions)", 10.0, 500.0, float(selected_row["monthly_retainer_idr"]), step=5.0)
                in_ad_spend = st.slider("Budget Fluctuation (%)", -50.0, 50.0, float(selected_row["ad_spend_fluctuation_pct"]), step=1.0)

        with st.expander("Operations & Billing Health", expanded=True):
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                in_delay = st.slider("Invoice Delay (Days)", 0.0, 60.0, float(selected_row["payment_delay_days_avg"]))
                in_unpaid = st.number_input("Unpaid Invoices", 0, 10, int(selected_row["unpaid_invoices_count"]))
            with col_o2:
                in_tickets = st.number_input("Quarterly Revision Tickets", 0, 50, int(selected_row["support_tickets_count"]))
                in_sla_breach = st.number_input("Critical SLA Breaches", 0, 10, int(selected_row["sla_breach_count"]))

        with st.expander("Stakeholder Engagement & CSAT", expanded=True):
            in_csat = st.slider("Stakeholder CSAT Score (1 - 10)", 1.0, 10.0, float(selected_row["client_satisfaction_score"]), step=0.1)
            in_turnover = st.selectbox(
                "Key PIC / Stakeholder Change?",
                options=[("No turnover", 0), ("Yes, primary lead replaced", 1)],
                index=int(selected_row["stakeholder_turnover"]),
                format_func=lambda x: x[0],
            )[1]
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                in_qbr = st.slider("QBR Attendance", 0.0, 1.0, float(selected_row["qbr_attendance_rate"]), step=0.05)
            with col_e2:
                in_scope_creep = st.slider("Scope Creep Index", 1.0, 5.0, float(selected_row["scope_creep_index"]), step=0.1)

    # Compute Single Inference
    single_dict = {
        "service_tier": in_service,
        "contract_duration_months": in_contract_dur,
        "tenure_months": in_tenure,
        "monthly_retainer_idr": in_mrr,
        "payment_delay_days_avg": in_delay,
        "unpaid_invoices_count": in_unpaid,
        "support_tickets_count": in_tickets,
        "sla_breach_count": in_sla_breach,
        "scope_creep_index": in_scope_creep,
        "stakeholder_turnover": in_turnover,
        "qbr_attendance_rate": in_qbr,
        "client_satisfaction_score": in_csat,
        "ad_spend_fluctuation_pct": in_ad_spend,
    }
    df_single = pd.DataFrame([single_dict])
    prob = pipeline.predict_proba(df_single)[0, 1]
    prob_pct = prob * 100

    with sim_col_right:
        st.markdown("##### Live Retention Assessment")
        
        if prob_pct < 35:
            badge_html = '<span class="badge-pill badge-safe">LOW RETENTION RISK</span>'
            status_color = "#10b981"
        elif prob_pct < 65:
            badge_html = '<span class="badge-pill badge-warning">MODERATE RISK</span>'
            status_color = "#f59e0b"
        else:
            badge_html = '<span class="badge-pill badge-critical">CRITICAL RETENTION RISK</span>'
            status_color = "#f43f5e"

        st.markdown(
            f"""<div class="glass-card" style="text-align: center; padding: 26px 20px; margin-bottom: 18px;">
                <div style="margin-bottom: 12px;">{badge_html}</div>
                <div style="font-size: 3.4rem; font-weight: 700; letter-spacing: -0.04em; color: {status_color}; line-height: 1;">
                    {prob_pct:.1f}%
                </div>
                <div style="font-size: 0.86rem; color: #94a3b8; margin-top: 8px;">
                    Estimated probability of non-renewal
                </div>
                <div style="margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.8rem; color: #64748b;">
                    Monthly Retainer: Rp {in_mrr:,.1f} M • Partnership Age: {in_tenure} Months
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("**Prescriptive Retention Interventions**")
        recs = get_retention_recommendations(single_dict, prob)
        for r in recs:
            st.markdown(f'<div class="action-box">{r}</div>', unsafe_allow_html=True)

    # Feature Attribution Breakdown with human-readable labels
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 28px 0 20px 0;'/>", unsafe_allow_html=True)
    st.markdown("##### Primary Risk Drivers (Explainable AI Attribution)")
    
    trained_preprocessor = pipeline.named_steps["preprocessor"]
    X_single_trans = trained_preprocessor.transform(df_single)
    feature_names = summary["feature_names"]
    shap_vals = explainer.shap_values(X_single_trans)
    
    if isinstance(shap_vals, list):
        row_shap = shap_vals[1][0]
    elif len(shap_vals.shape) == 2:
        row_shap = shap_vals[0]
    else:
        row_shap = shap_vals[0]

    human_features = [FEATURE_LABEL_MAP.get(f, f) for f in feature_names]
    shap_df = pd.DataFrame({"Predictive Factor": human_features, "Attribution": row_shap}).sort_values(by="Attribution", ascending=True)
    shap_df["Impact Direction"] = np.where(shap_df["Attribution"] > 0, "Elevates Churn Risk", "Protects / Stabilizes Account")

    fig_shap = px.bar(
        shap_df,
        x="Attribution",
        y="Predictive Factor",
        orientation="h",
        color="Impact Direction",
        color_discrete_map={
            "Elevates Churn Risk": "#f43f5e",
            "Protects / Stabilizes Account": "#10b981",
        },
    )
    layout_shap = get_apple_chart_layout("Factor Contribution Breakdown for this Account", height=380)
    fig_shap.update_layout(layout_shap)
    st.plotly_chart(fig_shap, width="stretch")

# ----------------- TAB 3: CLIENT DIRECTORY & ACTION LIST -----------------
with tab_directory:
    st.markdown("#### Contract Portfolio Directory")
    st.caption("Filter and prioritize client interventions by risk tier and revenue weight.")

    f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 1])
    with f_col1:
        search_query = st.text_input("Search Account ID", placeholder="e.g. ACC-1045")
    with f_col2:
        filter_tier = st.multiselect(
            "Service Division",
            options=df_scored["service_tier"].unique().tolist(),
            default=df_scored["service_tier"].unique().tolist(),
        )
    with f_col3:
        filter_risk = st.multiselect(
            "Risk Tier",
            options=["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk"],
        )
    with f_col4:
        filter_min_mrr = st.slider("Minimum Monthly Retainer (IDR Millions)", 15, 200, 15)

    filtered = df_scored[
        (df_scored["service_tier"].isin(filter_tier))
        & (df_scored["risk_tier"].isin(filter_risk))
        & (df_scored["monthly_retainer_idr"] >= filter_min_mrr)
    ]

    if search_query.strip():
        filtered = filtered[filtered["client_id"].str.contains(search_query.strip(), case=False, na=False)]

    filtered = filtered.sort_values(by="churn_probability", ascending=False)

    st.write(f"Displaying **{len(filtered)} accounts** matching active filters:")

    table_cols = [
        "client_id",
        "service_tier",
        "monthly_retainer_idr",
        "churn_probability",
        "risk_tier",
        "client_satisfaction_score",
        "payment_delay_days_avg",
        "sla_breach_count",
        "stakeholder_turnover",
    ]
    renamed_df = filtered[table_cols].rename(columns={
        "client_id": "Account ID",
        "service_tier": "Service Division",
        "monthly_retainer_idr": "MRR (IDR M)",
        "churn_probability": "Risk (%)",
        "risk_tier": "Risk Tier",
        "client_satisfaction_score": "CSAT (1-10)",
        "payment_delay_days_avg": "Delay (Days)",
        "sla_breach_count": "SLA Breaches",
        "stakeholder_turnover": "PIC Changed",
    })
    
    st.dataframe(renamed_df, width="stretch", hide_index=True)

    csv_export = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Accounts (CSV)",
        data=csv_export,
        file_name="enterprise_accounts_at_risk.csv",
        mime="text/csv",
    )

# ----------------- TAB 4: MODEL GOVERNANCE -----------------
with tab_model:
    st.markdown("#### Model Architecture & Validation Benchmark")
    st.markdown(
        f"Active production model: **{summary['best_model_name']}** • Test set **ROC-AUC: {summary['best_auc']:.4f}**."
    )

    metrics_df = (
        pd.DataFrame(summary["metrics_comparison"])
        .T.reset_index()
        .rename(columns={
            "index": "Model Architecture",
            "roc_auc": "ROC-AUC",
            "f1_score": "F1-Score",
            "precision": "Precision",
            "recall": "Recall",
            "accuracy": "Accuracy",
        })
    )
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Global Factor Importance Weights")

    trained_model = pipeline.named_steps["classifier"]
    if hasattr(trained_model, "feature_importances_"):
        raw_imp = trained_model.feature_importances_
    elif hasattr(trained_model, "coef_"):
        raw_imp = np.abs(trained_model.coef_[0])
    else:
        raw_imp = np.ones(len(summary["feature_names"]))

    human_global_features = [FEATURE_LABEL_MAP.get(f, f) for f in summary["feature_names"]]
    global_imp_df = pd.DataFrame({
        "Predictive Factor": human_global_features,
        "Relative Importance": raw_imp,
    }).sort_values(by="Relative Importance", ascending=True)

    fig_global = px.bar(
        global_imp_df,
        x="Relative Importance",
        y="Predictive Factor",
        orientation="h",
        color="Relative Importance",
        color_continuous_scale="Teal",
    )
    layout_global = get_apple_chart_layout("Global Predictive Factor Weights across Portfolio", height=400)
    fig_global.update_layout(layout_global)
    st.plotly_chart(fig_global, width="stretch")

st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 32px 0 16px 0;'/>", unsafe_allow_html=True)
st.caption("Enterprise Retention Intelligence Platform • Automated inference via scikit-learn & XAI.")
