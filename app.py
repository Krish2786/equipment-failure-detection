
import csv
import hashlib
import io
import sys
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the ML source importable no matter which directory Streamlit is launched
# from, then load the same predictor used by the FastAPI service.
PROJECT_ROOT = Path(__file__).resolve().parent
ML_SRC = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from model_prediction import EquipmentFailurePredictor
from batch_prediction import (
    MAX_BATCH_ROWS,
    build_batch_report,
    template_csv,
)
from database import (
    authenticate_user,
    clear_predictions,
    create_user,
    get_distinct_machines,
    get_fleet_summary,
    get_machine_trend,
    get_prediction_stats,
    get_predictions,
    init_db,
    normalize_machine_id,
    reassign_machine_id,
    save_prediction,
    save_predictions_bulk,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_table(headers, rows):
    """Render a table without Streamlit's Arrow-based dataframe serializer."""
    header_html = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    st.markdown(
        f"""
        <table class="app-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True
    )


def render_risk_distribution(risk_counts):
    """Render risk counts without Streamlit's Altair-based chart serializer."""
    maximum = max(risk_counts.values(), default=0)
    colors = {
        "LOW": "#4ade80",
        "MODERATE": "#facc15",
        "HIGH": "#fb923c",
        "CRITICAL": "#f87171",
    }
    bars = []
    for risk_level, count in risk_counts.items():
        width = (count / maximum * 100) if maximum else 0
        bars.append(
            f"""
            <div style="margin: 10px 0;">
                <div style="display:flex;justify-content:space-between;
                            color:#e2e8f0;font-size:13px;margin-bottom:4px;">
                    <span>{escape(str(risk_level))}</span>
                    <strong>{int(count):,}</strong>
                </div>
                <div style="background:#203956;border-radius:6px;height:12px;">
                    <div style="background:{colors.get(risk_level, '#67e8f9')};
                                border-radius:6px;height:12px;width:{width:.2f}%;"></div>
                </div>
            </div>
            """
        )
    st.markdown("".join(bars), unsafe_allow_html=True)


def render_asset_trend_chart(machine_id: str, trend_data: list[dict]):
    """Render dual-metric degradation trend charts for an individual asset."""
    if not trend_data:
        return

    x_labels = [
        f"#{r['id']} ({r['created_at'].split()[1][:5] if ' ' in r['created_at'] else r['created_at']})"
        for r in trend_data
    ]
    probabilities = [r["failure_percentage"] for r in trend_data]
    tool_wears = [r["tool_wear"] for r in trend_data]

    if PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=(
                f"Failure Probability (%) vs 35% Alert Threshold — Asset {machine_id}",
                f"Cutting Tool Wear Progression (min) — Asset {machine_id}",
            ),
        )

        # 1. Failure Probability Trace
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=probabilities,
                mode="lines+markers",
                name="Failure Risk (%)",
                line=dict(color="#ef4444", width=3),
                marker=dict(size=8, color="#dc2626"),
                customdata=[
                    [
                        r["created_at"],
                        r["risk_level"],
                        r["torque"],
                        r["rotational_speed"],
                    ]
                    for r in trend_data
                ],
                hovertemplate=(
                    "<b>Check %{x}</b><br>"
                    "Timestamp: %{customdata[0]}<br>"
                    "Failure Risk: <b>%{y:.2f}%</b><br>"
                    "Risk Level: %{customdata[1]}<br>"
                    "Torque: %{customdata[2]} Nm<br>"
                    "Speed: %{customdata[3]} RPM<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Horizontal 35% Decision Threshold Line
        fig.add_hline(
            y=35.0,
            line_dash="dash",
            line_color="#f59e0b",
            line_width=2.5,
            annotation_text="35.0% Decision Threshold (Maintenance Trigger)",
            annotation_position="top right",
            annotation_font=dict(size=12, color="#f59e0b"),
            row=1,
            col=1,
        )

        # 2. Tool Wear Trace
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=tool_wears,
                mode="lines+markers",
                name="Tool Wear (min)",
                line=dict(color="#38bdf8", width=3),
                marker=dict(size=8, color="#0284c7"),
                customdata=[[r["created_at"]] for r in trend_data],
                hovertemplate=(
                    "<b>Check %{x}</b><br>"
                    "Timestamp: %{customdata[0]}<br>"
                    "Tool Wear: <b>%{y:.0f} min</b><extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=500,
            margin=dict(l=50, r=40, t=50, b=40),
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(30, 41, 59, 0.5)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
        fig.update_yaxes(title_text="Wear (min)", row=2, col=1)
        fig.update_xaxes(title_text="Inspection Sequence / Check", row=2, col=1)
        try:
            st.plotly_chart(fig, width="stretch")
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Fallback using built-in Streamlit charts
        st.caption(f"Asset Degradation Trend: {machine_id}")
        chart_df = pd.DataFrame(
            {
                "Failure Risk (%)": probabilities,
                "35% Threshold": [35.0] * len(probabilities),
            },
            index=x_labels,
        )
        st.line_chart(chart_df)

        wear_df = pd.DataFrame({"Tool Wear (min)": tool_wears}, index=x_labels)
        st.line_chart(wear_df)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ML System for Early Detection of Equipment Failure",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background: #0c1628;
    color: #f8fafc;
}

.main .block-container {
    max-width: 1450px;
    padding: 2rem 2.5rem;
}

h1, h2, h3, h4, p, label {
    color: #f1f5f9;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #14243b;
    border-right: 1px solid #304965;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

[data-testid="stSidebar"] h1 {
    color: #67e8f9 !important;
}

/* PROJECT BRAND */

.sidebar-user {
    background: #1b3350;
    border: 1px solid #355575;
    border-radius: 10px;
    padding: 10px 12px;
    margin: 12px 0 14px 0;
}

.sidebar-user-name {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 700;
    line-height: 1.3;
}

.sidebar-user-email {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 2px;
    word-break: break-all;
}

.project-brand {
    font-size: 21px;
    font-weight: 850;
    line-height: 1.4;
    color: #67e8f9;
}

.project-subtitle {
    font-size: 11px;
    color: #94a3b8;
    letter-spacing: 1.5px;
}

/* TOP HEADER */

.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #172b46;
    border: 1px solid #355575;
    border-radius: 13px;
    padding: 15px 20px;
    margin-bottom: 23px;
}

.header-label {
    color: #e0f2fe;
    font-size: 14px;
    font-weight: 750;
}

.header-status {
    color: #4ade80;
    font-size: 12px;
    font-weight: 750;
}

.header-status-group {
    display: flex;
    align-items: center;
    gap: 18px;
}

/* HERO SECTION */

.hero {
    background: linear-gradient(115deg, #193b60, #17314e 65%, #16465a);
    border: 1px solid #3b668b;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 24px;
}

.hero-tag {
    color: #67e8f9;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 13px;
}

.hero-title {
    color: #ffffff;
    font-size: 32px;
    font-weight: 900;
    line-height: 1.3;
    margin-bottom: 12px;
}

.hero-description {
    color: #c6d9eb;
    font-size: 14px;
    line-height: 1.8;
}

.hero-icon {
    font-size: 72px;
    text-align: center;
    padding-top: 15px;
}

/* SECTION HEADINGS */

.section-heading {
    color: #67e8f9;
    font-size: 21px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 6px;
}

.section-description {
    color: #9fb2c8;
    font-size: 13px;
    margin-bottom: 17px;
}

/* OVERVIEW CARDS */

.overview-card {
    background: linear-gradient(145deg, #1c3452, #172940);
    border: 1px solid #355575;
    border-radius: 14px;
    padding: 18px;
    min-height: 125px;
    transition: 0.2s ease;
}

.overview-card:hover {
    border-color: #38bdf8;
    transform: translateY(-3px);
}

.overview-icon {
    font-size: 25px;
    margin-bottom: 9px;
}

.overview-label {
    color: #a9bfd5;
    font-size: 11px;
    font-weight: 700;
}

.overview-value {
    color: white;
    font-size: 20px;
    font-weight: 850;
    margin-top: 7px;
}

.overview-note {
    color: #67e8f9;
    font-size: 11px;
    margin-top: 5px;
}

/* SENSOR PANELS */

.sensor-panel {
    background: #172b45;
    border: 1px solid #355575;
    border-radius: 13px;
    padding: 16px;
    margin-top: 12px;
    margin-bottom: 12px;
}

.sensor-title {
    color: #67e8f9;
    font-size: 16px;
    font-weight: 800;
}

.sensor-description {
    color: #a9bfd5;
    font-size: 12px;
    margin-top: 5px;
}

/* INPUT FIELDS */

.stNumberInput input {
    background: #203956 !important;
    color: #ffffff !important;
    border: 1px solid #496987 !important;
    border-radius: 9px !important;
}

.stNumberInput input:focus {
    border-color: #38bdf8 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #203956;
    border-color: #496987;
    border-radius: 9px;
    color: white;
}

.stSelectbox [data-baseweb="select"] * {
    color: white !important;
}

/* FORM */

[data-testid="stForm"] {
    background: #14243b;
    border: 1px solid #355575;
    border-radius: 16px;
    padding: 22px;
}

/* BUTTONS */

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(100deg, #0891b2, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    height: 46px;
    font-weight: 800;
    transition: 0.2s;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    filter: brightness(1.2);
    transform: translateY(-2px);
}

/* METRICS */

[data-testid="stMetric"] {
    background: #1b304b;
    border: 1px solid #355575;
    border-radius: 13px;
    padding: 17px;
}

[data-testid="stMetricLabel"] {
    color: #b8cce0 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 850;
}

/* RESULT PANELS */

.success-result {
    background: linear-gradient(120deg, #123f35, #174d3d);
    border: 1px solid #22c55e;
    border-left: 5px solid #4ade80;
    border-radius: 14px;
    padding: 23px;
    margin-top: 15px;
}

.failure-result {
    background: linear-gradient(120deg, #492632, #542b39);
    border: 1px solid #fb7185;
    border-left: 5px solid #fb7185;
    border-radius: 14px;
    padding: 23px;
    margin-top: 15px;
}

.result-title {
    color: white;
    font-size: 24px;
    font-weight: 900;
}

.result-description {
    color: #e2e8f0;
    font-size: 14px;
    line-height: 1.7;
    margin-top: 8px;
}

/* RESULT CARDS */

.result-card {
    background: #172b45;
    border: 1px solid #355575;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
}

.result-label {
    color: #a9bfd5;
    font-size: 12px;
    font-weight: 700;
}

.result-value {
    color: #67e8f9;
    font-size: 29px;
    font-weight: 900;
    margin: 12px 0;
}

/* INFORMATION CARDS */

.info-card {
    background: #172b45;
    border: 1px solid #355575;
    border-radius: 13px;
    padding: 19px;
    min-height: 145px;
}

.info-card-title {
    color: #67e8f9;
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 12px;
}

.info-card-description {
    color: #c1d1e2;
    font-size: 13px;
    line-height: 1.9;
}

/* WORKFLOW */

.workflow-card {
    background: #172b45;
    border: 1px solid #355575;
    border-radius: 12px;
    padding: 17px;
    margin-bottom: 10px;
}

.workflow-number {
    color: #67e8f9;
    font-size: 20px;
    font-weight: 900;
}

.workflow-title {
    color: white;
    font-size: 15px;
    font-weight: 800;
    margin-top: 5px;
}

.workflow-description {
    color: #a9bfd5;
    font-size: 13px;
    margin-top: 5px;
}

/* TABLES */

[data-testid="stDataFrame"] {
    border: 1px solid #355575;
    border-radius: 12px;
}

/* EXPANDER */

[data-testid="stExpander"] {
    background: #172b45;
    border: 1px solid #355575;
    border-radius: 12px;
}

/* FOOTER */

.footer {
    text-align: center;
    padding: 22px;
    color: #94a3b8;
    font-size: 12px;
    border-top: 1px solid #304965;
    margin-top: 30px;
}

.footer-title {
    color: #67e8f9;
    font-weight: 800;
    font-size: 14px;
}

hr {
    border-color: #304965 !important;
}

/* FLEET OVERVIEW CARDS */

.fleet-card {
    background: linear-gradient(145deg, #1c3452, #172940);
    border: 1px solid #355575;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: 0.2s ease;
    position: relative;
}

.fleet-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
    box-shadow: 0 4px 18px rgba(56, 189, 248, 0.15);
}

.fleet-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.fleet-machine-id {
    color: #67e8f9;
    font-size: 17px;
    font-weight: 900;
    letter-spacing: 0.5px;
}

.fleet-risk-badge {
    font-size: 11px;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
}

.fleet-risk-LOW    { background: #14532d; color: #4ade80; border: 1px solid #22c55e; }
.fleet-risk-MODERATE { background: #422006; color: #fbbf24; border: 1px solid #f59e0b; }
.fleet-risk-HIGH   { background: #431407; color: #fb923c; border: 1px solid #ea580c; }
.fleet-risk-CRITICAL { background: #450a0a; color: #f87171; border: 1px solid #ef4444; }
.fleet-risk-UNKNOWN  { background: #1e293b; color: #94a3b8; border: 1px solid #475569; }

.fleet-trend-arrow {
    font-size: 20px;
    font-weight: 900;
}

.fleet-stat-row {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    margin-top: 6px;
}

.fleet-stat {
    display: flex;
    flex-direction: column;
    min-width: 70px;
}

.fleet-stat-label {
    color: #7a9cb8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

.fleet-stat-value {
    color: #e2e8f0;
    font-size: 15px;
    font-weight: 800;
    margin-top: 2px;
}

.fleet-last-check {
    color: #64748b;
    font-size: 11px;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stDeployButton"],
    [data-testid="stToolbar"] button[title="Deploy"],
    [data-testid="stToolbar"] a[title="Deploy"],
    [data-testid="stToolbar"] [aria-label="Deploy"],
    button[title="Deploy"],
    a[title="Deploy"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        font-size: 15px !important;
        font-weight: 600 !important;
        min-height: 34px !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# AUTHENTICATION
# =========================================================

init_db()

SESSION_DURATION = timedelta(hours=1)


def start_user_session(user):
    """Store the authenticated user and expire the session after one hour."""
    st.session_state["user"] = user
    st.session_state["session_expires_at"] = (
        datetime.now().astimezone() + SESSION_DURATION
    ).isoformat()


def render_auth_page():
    """Login / signup screen shown whenever no user is signed in."""

    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">SECURE ACCESS</div>'
        '<div class="hero-title">MachineGuard</div>'
        '<div class="hero-description">'
        'Sign in or create an account to run equipment failure predictions '
        'and keep a personal history of every analysis you perform.'
        '</div></div>',
        unsafe_allow_html=True
    )

    _, center, _ = st.columns([1, 1.4, 1])

    with center:

        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        with login_tab:

            with st.form("login_form"):

                email = st.text_input(
                    "Email address",
                    key="login_email",
                    placeholder="you@example.com"
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                    placeholder="Your password"
                )

                login_submitted = st.form_submit_button(
                    "Login",
                    use_container_width=True
                )

            if login_submitted:

                if not email.strip() or not password:
                    st.error("Please enter both your email and password.")

                else:
                    user = authenticate_user(email, password)

                    if user is None:
                        st.error("Invalid email or password. Please try again.")

                    else:
                        start_user_session(user)
                        st.session_state["flash"] = f"Welcome back, {user['name']}!"
                        st.rerun()

        with signup_tab:

            with st.form("signup_form"):

                name = st.text_input(
                    "Full name",
                    key="signup_name",
                    placeholder="Jane Doe"
                )

                email = st.text_input(
                    "Email address",
                    key="signup_email",
                    placeholder="you@example.com"
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="signup_password",
                    help="At least 8 characters"
                )

                confirm = st.text_input(
                    "Confirm password",
                    type="password",
                    key="signup_confirm"
                )

                signup_submitted = st.form_submit_button(
                    "Create account",
                    use_container_width=True
                )

            if signup_submitted:

                if password != confirm:
                    st.error("The two passwords do not match.")

                else:
                    try:
                        user = create_user(name, email, password)

                    except ValueError as exc:
                        st.error(str(exc))

                    else:
                        start_user_session(user)
                        st.session_state["flash"] = (
                            f"Account created. Welcome, {user['name']}!"
                        )
                        st.rerun()

        st.caption(
            "Accounts are stored locally in data/machineguard.db. "
            "Passwords are hashed with PBKDF2-SHA256."
        )


def render_fleet_page(user_id: int) -> None:
    """Analytical Fleet Overview Dashboard — per-machine cards, charts, trend arrows."""

    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">FLEET HEALTH INTELLIGENCE</div>'
        '<div class="hero-title">Fleet Overview Dashboard</div>'
        '<div class="hero-description">'
        'A real-time analytical view of every tracked machine asset. '
        'Each card shows the latest risk level, trend direction, failure probability, '
        'and historical charts — so you can prioritise maintenance across the entire fleet.'
        '</div></div>',
        unsafe_allow_html=True
    )

    fleet = get_fleet_summary(user_id)

    if not fleet:
        st.info(
            "No tracked machines yet. Run a prediction on the **Prediction Dashboard** "
            "or upload a CSV on **Batch Prediction** to start building your fleet."
        )
        return

    # ── Fleet-level KPI bar ──────────────────────────────────────────────────
    total_machines = len(fleet)
    critical_count = sum(1 for m in fleet if m["latest_risk"] in ("CRITICAL", "HIGH"))
    avg_risk = sum(m["latest_failure_pct"] for m in fleet) / total_machines
    total_checks = sum(m["total_checks"] for m in fleet)
    machines_above_threshold = sum(1 for m in fleet if m["latest_failure_pct"] >= 35.0)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Machines Tracked", total_machines)
    k2.metric("HIGH / CRITICAL Now", critical_count)
    k3.metric("Above Threshold (≥35%)", machines_above_threshold)
    k4.metric("Fleet Avg Risk", f"{avg_risk:.1f}%")
    k5.metric("Total Checks Logged", f"{total_checks:,}")

    st.markdown("---")

    # ── Sort / filter controls ───────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">Fleet Sort &amp; Filter</div>',
        unsafe_allow_html=True
    )
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        sort_by = st.selectbox(
            "Sort by",
            ["Latest Risk % (High→Low)", "Machine ID (A→Z)", "Total Checks (High→Low)",
             "Peak Risk (High→Low)", "Last Check (Newest)"],
            key="fleet_sort"
        )
    with ctrl2:
        risk_filter_fleet = st.selectbox(
            "Filter by Risk Level",
            ["All", "CRITICAL", "HIGH", "MODERATE", "LOW"],
            key="fleet_risk_filter"
        )
    with ctrl3:
        cards_per_row = st.selectbox("Cards per row", [2, 3, 4], index=1, key="fleet_cols")

    # apply filter
    visible_fleet = (
        fleet if risk_filter_fleet == "All"
        else [m for m in fleet if m["latest_risk"] == risk_filter_fleet]
    )

    # apply sort
    if sort_by == "Machine ID (A→Z)":
        visible_fleet = sorted(visible_fleet, key=lambda m: m["machine_id"])
    elif sort_by == "Total Checks (High→Low)":
        visible_fleet = sorted(visible_fleet, key=lambda m: m["total_checks"], reverse=True)
    elif sort_by == "Peak Risk (High→Low)":
        visible_fleet = sorted(visible_fleet, key=lambda m: m["peak_failure_pct"], reverse=True)
    elif sort_by == "Last Check (Newest)":
        visible_fleet = sorted(visible_fleet, key=lambda m: m["last_check_at"], reverse=True)
    else:
        visible_fleet = sorted(visible_fleet, key=lambda m: m["latest_failure_pct"], reverse=True)

    if not visible_fleet:
        st.info("No machines match the selected risk filter.")
        return

    st.markdown("---")
    st.markdown(
        f'<div class="section-heading">Machine Cards ({len(visible_fleet)} shown)</div>'
        '<div class="section-description">Each card shows live risk badge, trend arrow, '
        'and key sensor readings. Click on a machine to see its full degradation chart below.</div>',
        unsafe_allow_html=True
    )

    # ── Helper: trend arrow & colour ────────────────────────────────────────
    def _trend(first_pct: float, latest_pct: float) -> tuple[str, str, str]:
        """Returns (arrow_emoji, colour, label)."""
        delta = latest_pct - first_pct
        if abs(delta) < 2.0:
            return "➡️", "#94a3b8", f"Stable ({delta:+.1f}%)"
        if delta > 0:
            return "📈", "#f87171", f"Worsening ({delta:+.1f}%)"
        return "📉", "#4ade80", f"Improving ({delta:+.1f}%)"

    # ── Machine cards grid ───────────────────────────────────────────────────
    _risk_badge_class = {
        "LOW": "fleet-risk-LOW",
        "MODERATE": "fleet-risk-MODERATE",
        "HIGH": "fleet-risk-HIGH",
        "CRITICAL": "fleet-risk-CRITICAL",
    }

    for row_start in range(0, len(visible_fleet), cards_per_row):
        cols = st.columns(cards_per_row)
        for col_idx, machine in enumerate(visible_fleet[row_start: row_start + cards_per_row]):
            arrow, arrow_color, arrow_label = _trend(
                machine["first_failure_pct"], machine["latest_failure_pct"]
            )
            badge_class = _risk_badge_class.get(machine["latest_risk"], "fleet-risk-UNKNOWN")
            failure_rate = (
                machine["failure_checks"] / machine["total_checks"] * 100
                if machine["total_checks"] > 0 else 0.0
            )
            last_check_short = machine["last_check_at"][:16] if machine["last_check_at"] else "—"

            with cols[col_idx]:
                st.markdown(
                    f'<div class="fleet-card">'
                    f'  <div class="fleet-card-header">'
                    f'    <span class="fleet-machine-id">⚙️ {escape(machine["machine_id"])}</span>'
                    f'    <span class="fleet-risk-badge {badge_class}">{machine["latest_risk"]}</span>'
                    f'  </div>'
                    f'  <div class="fleet-stat-row">'
                    f'    <div class="fleet-stat">'
                    f'      <span class="fleet-stat-label">Latest Risk</span>'
                    f'      <span class="fleet-stat-value" style="color:#67e8f9">'
                    f'        {machine["latest_failure_pct"]:.1f}%'
                    f'      </span>'
                    f'    </div>'
                    f'    <div class="fleet-stat">'
                    f'      <span class="fleet-stat-label">Peak Risk</span>'
                    f'      <span class="fleet-stat-value" style="color:#fb923c">'
                    f'        {machine["peak_failure_pct"]:.1f}%'
                    f'      </span>'
                    f'    </div>'
                    f'    <div class="fleet-stat">'
                    f'      <span class="fleet-stat-label">Avg Risk</span>'
                    f'      <span class="fleet-stat-value">{machine["avg_failure_pct"]:.1f}%</span>'
                    f'    </div>'
                    f'    <div class="fleet-stat">'
                    f'      <span class="fleet-stat-label">Checks</span>'
                    f'      <span class="fleet-stat-value">{machine["total_checks"]}</span>'
                    f'    </div>'
                    f'    <div class="fleet-stat">'
                    f'      <span class="fleet-stat-label">Failure Rate</span>'
                    f'      <span class="fleet-stat-value">{failure_rate:.0f}%</span>'
                    f'    </div>'
                    f'    <div class="fleet-stat">'
                    f'      <span class="fleet-stat-label">Tool Wear</span>'
                    f'      <span class="fleet-stat-value">{machine["latest_tool_wear"]:.0f} min</span>'
                    f'    </div>'
                    f'  </div>'
                    f'  <div style="margin-top:10px; font-size:14px;">'
                    f'    <span class="fleet-trend-arrow">{arrow}</span>'
                    f'    <span style="color:{arrow_color}; font-size:12px; margin-left:6px;">'
                    f'      {escape(arrow_label)}'
                    f'    </span>'
                    f'  </div>'
                    f'  <div class="fleet-last-check">🕐 Last check: {escape(last_check_short)}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Fleet-wide Plotly charts ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">📊 Fleet-Wide Risk Analytics</div>'
        '<div class="section-description">'
        'Comparative charts across all tracked machines — spot outliers and prioritise inspections.'
        '</div>',
        unsafe_allow_html=True
    )

    machine_ids = [m["machine_id"] for m in visible_fleet]
    latest_risks = [m["latest_failure_pct"] for m in visible_fleet]
    avg_risks = [m["avg_failure_pct"] for m in visible_fleet]
    peak_risks = [m["peak_failure_pct"] for m in visible_fleet]
    tool_wears = [m["latest_tool_wear"] for m in visible_fleet]

    # Colour each bar by risk level
    _risk_color = {
        "LOW": "#4ade80",
        "MODERATE": "#facc15",
        "HIGH": "#fb923c",
        "CRITICAL": "#f87171",
        "UNKNOWN": "#94a3b8",
    }
    bar_colors = [_risk_color.get(m["latest_risk"], "#94a3b8") for m in visible_fleet]

    if PLOTLY_AVAILABLE:
        # Chart 1 — Latest vs Average vs Peak risk comparison
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown(
                '<div class="section-description">Latest vs Average vs Peak Failure Risk (%)</div>',
                unsafe_allow_html=True
            )
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name="Latest Risk",
                x=machine_ids,
                y=latest_risks,
                marker_color=bar_colors,
                hovertemplate="<b>%{x}</b><br>Latest: %{y:.1f}%<extra></extra>",
            ))
            fig_cmp.add_trace(go.Scatter(
                name="Avg Risk",
                x=machine_ids,
                y=avg_risks,
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.5, dash="dot"),
                marker=dict(size=8, color="#38bdf8"),
                hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f}%<extra></extra>",
            ))
            fig_cmp.add_trace(go.Scatter(
                name="Peak Risk",
                x=machine_ids,
                y=peak_risks,
                mode="markers",
                marker=dict(size=12, color="#f59e0b", symbol="diamond"),
                hovertemplate="<b>%{x}</b><br>Peak: %{y:.1f}%<extra></extra>",
            ))
            fig_cmp.add_hline(
                y=35.0,
                line_dash="dash",
                line_color="#f59e0b",
                line_width=2,
                annotation_text="35% Alert Threshold",
                annotation_position="top right",
                annotation_font=dict(size=11, color="#f59e0b"),
            )
            fig_cmp.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(15,23,42,0.4)",
                plot_bgcolor="rgba(30,41,59,0.5)",
                margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                xaxis_title="Machine ID",
                yaxis_title="Failure Risk (%)",
            )
            try:
                st.plotly_chart(fig_cmp, width="stretch")
            except TypeError:
                st.plotly_chart(fig_cmp, use_container_width=True)

        # Chart 2 — Tool wear bar chart
        with chart2:
            st.markdown(
                '<div class="section-description">Current Tool Wear (min) per Machine</div>',
                unsafe_allow_html=True
            )
            fig_wear = go.Figure(go.Bar(
                x=machine_ids,
                y=tool_wears,
                marker_color="#38bdf8",
                marker_line_color="#0284c7",
                marker_line_width=1.5,
                hovertemplate="<b>%{x}</b><br>Tool Wear: %{y:.0f} min<extra></extra>",
            ))
            fig_wear.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(15,23,42,0.4)",
                plot_bgcolor="rgba(30,41,59,0.5)",
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis_title="Machine ID",
                yaxis_title="Tool Wear (min)",
            )
            try:
                st.plotly_chart(fig_wear, width="stretch")
            except TypeError:
                st.plotly_chart(fig_wear, use_container_width=True)

        # Chart 3 — Risk distribution pie
        st.markdown(
            '<div class="section-description" style="margin-top:16px;">Fleet Risk Level Distribution</div>',
            unsafe_allow_html=True
        )
        risk_distribution: dict[str, int] = {}
        for m in visible_fleet:
            risk_distribution[m["latest_risk"]] = risk_distribution.get(m["latest_risk"], 0) + 1

        pie_col, scatter_col = st.columns(2)

        with pie_col:
            pie_labels = list(risk_distribution.keys())
            pie_values = list(risk_distribution.values())
            pie_colors = [_risk_color.get(r, "#94a3b8") for r in pie_labels]
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker=dict(colors=pie_colors, line=dict(color="#0f172a", width=2)),
                hole=0.4,
                hovertemplate="<b>%{label}</b><br>%{value} machines (%{percent})<extra></extra>",
                textinfo="label+value",
            ))
            fig_pie.update_layout(
                height=320,
                template="plotly_dark",
                paper_bgcolor="rgba(15,23,42,0.4)",
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            try:
                st.plotly_chart(fig_pie, width="stretch")
            except TypeError:
                st.plotly_chart(fig_pie, use_container_width=True)

        # Chart 4 — Scatter: tool wear vs latest failure risk (bubble = checks)
        with scatter_col:
            st.markdown(
                '<span style="color:#9fb2c8;font-size:13px;">'
                'Tool Wear vs Failure Risk — bubble size = number of checks'
                '</span>',
                unsafe_allow_html=True
            )
            bubble_sizes = [max(m["total_checks"] * 6, 10) for m in visible_fleet]
            fig_scatter = go.Figure(go.Scatter(
                x=tool_wears,
                y=latest_risks,
                mode="markers+text",
                text=machine_ids,
                textposition="top center",
                textfont=dict(size=10, color="#e2e8f0"),
                marker=dict(
                    size=bubble_sizes,
                    color=latest_risks,
                    colorscale=[
                        [0.0, "#4ade80"],
                        [0.35, "#facc15"],
                        [0.6, "#fb923c"],
                        [1.0, "#f87171"],
                    ],
                    cmin=0,
                    cmax=100,
                    showscale=True,
                    colorbar=dict(title="Risk %", thickness=12, len=0.8),
                    line=dict(width=1, color="#0f172a"),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Tool Wear: %{x:.0f} min<br>"
                    "Failure Risk: %{y:.1f}%<extra></extra>"
                ),
            ))
            fig_scatter.add_hline(
                y=35.0,
                line_dash="dash",
                line_color="#f59e0b",
                line_width=1.8,
                annotation_text="35% threshold",
                annotation_font=dict(size=10, color="#f59e0b"),
            )
            fig_scatter.update_layout(
                height=320,
                template="plotly_dark",
                paper_bgcolor="rgba(15,23,42,0.4)",
                plot_bgcolor="rgba(30,41,59,0.5)",
                margin=dict(l=50, r=60, t=10, b=40),
                xaxis_title="Tool Wear (min)",
                yaxis_title="Failure Risk (%)",
            )
            try:
                st.plotly_chart(fig_scatter, width="stretch")
            except TypeError:
                st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        # Fallback if Plotly is unavailable
        st.caption("Install plotly>=5.18.0 for interactive charts.")
        df_fleet = pd.DataFrame({
            "Machine": machine_ids,
            "Latest Risk (%)": latest_risks,
            "Avg Risk (%)": avg_risks,
            "Tool Wear (min)": tool_wears,
        }).set_index("Machine")
        st.bar_chart(df_fleet[["Latest Risk (%)", "Avg Risk (%)"]])

    # ── Per-machine degradation chart selector ───────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">📈 Individual Asset Degradation Deep-Dive</div>'
        '<div class="section-description">'
        'Select any machine to view its full chronological failure probability '
        'and tool wear trend charts.'
        '</div>',
        unsafe_allow_html=True
    )

    deep_col, info_col = st.columns([1, 2])
    with deep_col:
        selected_machine = st.selectbox(
            "Select Machine for Deep-Dive",
            machine_ids,
            key="fleet_deep_machine"
        )
    with info_col:
        for m in visible_fleet:
            if m["machine_id"] == selected_machine:
                arrow, arrow_color, arrow_label = _trend(m["first_failure_pct"], m["latest_failure_pct"])
                st.markdown(
                    f'<div style="margin-top:28px;">'
                    f'<span class="fleet-risk-badge {_risk_badge_class.get(m["latest_risk"], "fleet-risk-UNKNOWN")}">'
                    f'{m["latest_risk"]}</span>&nbsp;&nbsp;'
                    f'<span style="color:{arrow_color};font-weight:800;">{arrow} {escape(arrow_label)}</span>'
                    f'&nbsp;&nbsp;<span style="color:#64748b;font-size:12px;">— {m["total_checks"]} checks logged</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                break

    trend_data = get_machine_trend(user_id, selected_machine)
    if not trend_data:
        st.info(f"No trend data found for **{selected_machine}**.")
    else:
        tc1, tc2, tc3, tc4 = st.columns(4)
        first_r, latest_r = trend_data[0], trend_data[-1]
        risk_delta_deep = latest_r["failure_percentage"] - first_r["failure_percentage"]
        wear_delta_deep = latest_r["tool_wear"] - first_r["tool_wear"]
        tc1.metric("Checks Logged", len(trend_data))
        tc2.metric(
            "Latest Risk",
            f"{latest_r['failure_percentage']:.1f}%",
            delta=f"{risk_delta_deep:+.1f}% vs first" if len(trend_data) > 1 else None,
            delta_color="inverse"
        )
        tc3.metric(
            "Tool Wear",
            f"{latest_r['tool_wear']:.0f} min",
            delta=f"{wear_delta_deep:+.0f} min" if len(trend_data) > 1 else None,
            delta_color="inverse"
        )
        tc4.metric("Peak Risk", f"{max(r['failure_percentage'] for r in trend_data):.1f}%")

        render_asset_trend_chart(selected_machine, trend_data)

    # ── Summary table (sortable) ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">📋 Fleet Summary Table</div>',
        unsafe_allow_html=True
    )
    render_table(
        ["Machine ID", "Risk Level", "Latest Risk %", "Avg Risk %", "Peak Risk %",
         "Checks", "Failures", "Tool Wear (min)", "Last Check"],
        [
            (
                m["machine_id"],
                m["latest_risk"],
                f"{m['latest_failure_pct']:.1f}%",
                f"{m['avg_failure_pct']:.1f}%",
                f"{m['peak_failure_pct']:.1f}%",
                m["total_checks"],
                m["failure_checks"],
                f"{m['latest_tool_wear']:.0f}",
                m["last_check_at"][:16] if m["last_check_at"] else "—",
            )
            for m in visible_fleet
        ]
    )


def render_history_page(user_id):
    """Per-user prediction history with filters, details, comparison and CSV export."""

    user = st.session_state["user"]
    predictions = get_predictions(user_id)

    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">PERSONAL AUDIT TRAIL</div>'
        '<div class="hero-title">Prediction History</div>'
        '<div class="hero-description">'
        f'All predictions made by <b>{escape(user["name"])}</b> are stored here. '
        'Filter the records, inspect a prediction, or compare two of them.'
        '</div></div>',
        unsafe_allow_html=True
    )

    if not predictions:
        st.info(
            "You have not run any predictions yet. Open "
            "**Prediction Dashboard** in the sidebar to create your first one."
        )
        return

    stats = get_prediction_stats(user_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", stats["total"])
    c2.metric("Failure Alerts", stats["failures"])
    c3.metric("Average Risk", f'{stats["avg_percentage"]:.1f}%')
    c4.metric("Highest Risk", f'{stats["max_percentage"]:.1f}%')

    st.markdown(
        '<div class="section-heading">Filter Records</div>'
        '<div class="section-description">'
        'Narrow the history by machine asset, risk level, outcome, or number of rows.'
        '</div>',
        unsafe_allow_html=True
    )

    distinct_machines = get_distinct_machines(user_id)
    all_machine_ids = sorted(
        list(
            {p.get("machine_id", "UNASSIGNED") for p in predictions if p.get("machine_id")}
            | set(distinct_machines)
        )
    )
    if not all_machine_ids:
        all_machine_ids = ["UNASSIGNED"]

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        machine_filter = st.selectbox(
            "Machine ID",
            ["All"] + all_machine_ids,
            key="history_machine_filter"
        )

    with f2:
        risk_filter = st.selectbox(
            "Risk level",
            ["All", "LOW", "MODERATE", "HIGH", "CRITICAL"],
            key="history_risk_filter"
        )

    with f3:
        outcome_filter = st.selectbox(
            "Outcome",
            ["All", "Failure", "Normal"],
            key="history_outcome_filter"
        )

    with f4:
        limit_choice = st.selectbox(
            "Rows to show",
            [10, 25, 50, 100, "All"],
            index=1,
            key="history_limit"
        )

    filtered = predictions

    if machine_filter != "All":
        filtered = [p for p in filtered if p.get("machine_id") == machine_filter]

    if risk_filter != "All":
        filtered = [p for p in filtered if p["risk_level"] == risk_filter]

    if outcome_filter != "All":
        wanted = 1 if outcome_filter == "Failure" else 0
        filtered = [p for p in filtered if p["prediction_code"] == wanted]

    if limit_choice != "All":
        filtered = filtered[:int(limit_choice)]

    st.markdown(
        f'<div class="section-heading">Records ({len(filtered)} shown)</div>',
        unsafe_allow_html=True
    )

    render_table(
        [
            "ID",
            "Date & Time",
            "Machine ID",
            "Scenario",
            "Failure Probability",
            "Risk Level",
            "Outcome"
        ],
        [
            [
                p["id"],
                p["created_at"],
                p.get("machine_id", "UNASSIGNED"),
                p["scenario"] or "Custom Input",
                f'{p["failure_percentage"]:.1f}%',
                p["risk_level"],
                "Failure" if p["prediction_code"] == 1 else "Normal"
            ]
            for p in filtered
        ]
    )

    # UNASSIGNED ASSET REASSIGNMENT SECTION
    unassigned_records = [p for p in predictions if p.get("machine_id") == "UNASSIGNED"]
    if unassigned_records:
        st.caption(
            "Records labeled UNASSIGNED were logged before Asset ID tracking was enabled or uploaded without a Machine ID."
        )
        with st.expander("Assign Machine ID to UNASSIGNED Records"):
            st.caption(
                f"You have **{len(unassigned_records)}** record(s) labeled as `UNASSIGNED`. "
                "Assign them to a real Machine ID to track their degradation trends and include them in Fleet Overview."
            )
            reassign_mode = st.radio(
                "Records to reassign",
                ["All UNASSIGNED records", "Specific record IDs"],
                horizontal=True,
                key="reassign_mode"
            )

            chosen_ids = None
            if reassign_mode == "Specific record IDs":
                id_choices = [p["id"] for p in unassigned_records]
                chosen_ids = st.multiselect(
                    "Select specific record IDs",
                    options=id_choices,
                    format_func=lambda pid: f"#{pid} ({next(p['created_at'] for p in unassigned_records if p['id'] == pid)} - {next(p['failure_percentage'] for p in unassigned_records if p['id'] == pid):.1f}%)",
                    key="reassign_multiselect"
                )

            reassign_c1, reassign_c2 = st.columns([2, 1])
            with reassign_c1:
                target_mid = st.text_input(
                    "Target Machine ID",
                    placeholder="e.g. MILL-01, LATHE-02",
                    help="Enter the unique Asset ID to assign to these records",
                    key="reassign_target_mid"
                )
            with reassign_c2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                btn_reassign = st.button("Assign Asset ID", key="btn_reassign_asset", use_container_width=True)

            if btn_reassign:
                norm_new_id = normalize_machine_id(target_mid)
                if not target_mid.strip() or norm_new_id == "UNASSIGNED":
                    st.error("Please enter a valid, non-empty Machine ID.")
                elif reassign_mode == "Specific record IDs" and not chosen_ids:
                    st.error("Please select at least one record ID to reassign.")
                else:
                    updated_count = reassign_machine_id(
                        user_id=user_id,
                        old_machine_id="UNASSIGNED",
                        new_machine_id=norm_new_id,
                        prediction_ids=chosen_ids if reassign_mode == "Specific record IDs" else None
                    )
                    st.session_state["flash"] = f"Successfully assigned {updated_count} record(s) to machine '{norm_new_id}'."
                    st.rerun()

    # ASSET DEGRADATION TREND CHARTS
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">📈 Asset Degradation & Telemetry Trend</div>'
        '<div class="section-description">'
        'Track failure probability escalation and cutting tool wear over time for individual equipment assets.'
        '</div>',
        unsafe_allow_html=True
    )

    target_machine = (
        machine_filter
        if machine_filter != "All"
        else (all_machine_ids[0] if all_machine_ids else "UNASSIGNED")
    )

    col_trend_sel, col_trend_info = st.columns([1, 2])
    with col_trend_sel:
        selected_trend_machine = st.selectbox(
            "Select Asset for Trend Analysis",
            all_machine_ids,
            index=all_machine_ids.index(target_machine) if target_machine in all_machine_ids else 0,
            key="trend_machine_select"
        )
    with col_trend_info:
        st.caption(
            "Visualizes chronological failure risk against the 35% operational alert threshold "
            "and cutting tool wear progression."
        )

    trend_records = get_machine_trend(user_id, selected_trend_machine)
    if not trend_records:
        st.info(f"No telemetry check history found for machine **{selected_trend_machine}**.")
    else:
        total_checks = len(trend_records)
        first_check = trend_records[0]
        latest_check = trend_records[-1]

        risk_delta = latest_check["failure_percentage"] - first_check["failure_percentage"]
        wear_delta = latest_check["tool_wear"] - first_check["tool_wear"]
        peak_risk = max(r["failure_percentage"] for r in trend_records)

        tm1, tm2, tm3, tm4 = st.columns(4)
        tm1.metric("Total Checks Logged", total_checks)
        tm2.metric(
            "Latest Failure Risk",
            f"{latest_check['failure_percentage']:.1f}%",
            delta=f"{risk_delta:+.1f}% vs first" if total_checks > 1 else None,
            delta_color="inverse"
        )
        tm3.metric(
            "Current Tool Wear",
            f"{latest_check['tool_wear']:.0f} min",
            delta=f"{wear_delta:+.0f} min" if total_checks > 1 else None,
            delta_color="inverse"
        )
        tm4.metric("Peak Risk Observed", f"{peak_risk:.1f}%")

        render_asset_trend_chart(selected_trend_machine, trend_records)

    # CSV EXPORT

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow([
        "ID", "Date & Time", "Machine ID", "Scenario",
        "Air Temperature (K)", "Process Temperature (K)",
        "Rotational Speed (RPM)", "Torque (Nm)", "Tool Wear (Minutes)",
        "Failure Probability (%)", "Risk Level", "Prediction Code",
        "Decision Threshold", "Recommended Action", "Model"
    ])

    for p in predictions:
        writer.writerow([
            p["id"],
            p["created_at"],
            p.get("machine_id", "UNASSIGNED"),
            p["scenario"],
            p["air_temperature"],
            p["process_temperature"],
            p["rotational_speed"],
            p["torque"],
            p["tool_wear"],
            p["failure_percentage"],
            p["risk_level"],
            p["prediction_code"],
            p["decision_threshold"],
            p["recommended_action"],
            p["model_name"]
        ])

    st.download_button(
        "Download my history (CSV)",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name=f"machineguard_history_user_{user_id}.csv",
        mime="text/csv",
        use_container_width=True,
        key="history_csv_download"
    )

    # DETAIL VIEW

    st.markdown(
        '<div class="section-heading">Inspect a Prediction</div>',
        unsafe_allow_html=True
    )

    records_by_label = {
        f'#{p["id"]} · {p.get("machine_id", "UNASSIGNED")} · {p["created_at"]} · {p["failure_percentage"]:.1f}% · {p["risk_level"]}': p
        for p in predictions
    }

    selected_label = st.selectbox(
        "Select a record",
        list(records_by_label.keys()),
        key="history_detail_select"
    )

    record = records_by_label[selected_label]

    d0, d1, d2, d3 = st.columns(4)
    d0.metric("Machine ID", record.get("machine_id", "UNASSIGNED"))
    d1.metric("Failure Probability", f'{record["failure_percentage"]:.2f}%')
    d2.metric("Risk Level", record["risk_level"])
    d3.metric("Scenario", record["scenario"] or "Custom Input")

    action = record["recommended_action"] or "No action recorded."

    if record["risk_level"] in ["HIGH", "CRITICAL"]:
        st.error(action)

    elif record["risk_level"] == "MODERATE":
        st.warning(action)

    else:
        st.success(action)

    render_table(
        ["Sensor Parameter", "Reading"],
        [
            ("Machine / Asset ID", record.get("machine_id", "UNASSIGNED")),
            ("Air Temperature (K)", f'{record["air_temperature"]:.1f}'),
            ("Process Temperature (K)", f'{record["process_temperature"]:.1f}'),
            ("Rotational Speed (RPM)", f'{record["rotational_speed"]:.0f}'),
            ("Torque (Nm)", f'{record["torque"]:.1f}'),
            ("Tool Wear (Minutes)", f'{record["tool_wear"]:.0f}'),
            ("Decision Threshold", f'{record["decision_threshold"] * 100:.0f}%'),
            ("Model", record["model_name"] or "Random Forest")
        ]
    )

    # COMPARISON VIEW

    st.markdown(
        '<div class="section-heading">Compare Two Predictions</div>'
        '<div class="section-description">'
        'Compare probability, risk and sensor values between two records.'
        '</div>',
        unsafe_allow_html=True
    )

    if len(predictions) < 2:
        st.caption("Run at least two predictions to use the comparison view.")

    else:
        labels = list(records_by_label.keys())

        left_col, right_col = st.columns(2)

        with left_col:
            a_label = st.selectbox(
                "Prediction A",
                labels,
                index=1,
                key="compare_a"
            )

        with right_col:
            b_label = st.selectbox(
                "Prediction B",
                labels,
                index=0,
                key="compare_b"
            )

        a = records_by_label[a_label]
        b = records_by_label[b_label]
        delta = b["failure_percentage"] - a["failure_percentage"]

        render_table(
            ["Field", "Prediction A", "Prediction B", "Difference"],
            [
                (
                    "Machine / Asset ID",
                    a.get("machine_id", "UNASSIGNED"),
                    b.get("machine_id", "UNASSIGNED"),
                    "Same" if a.get("machine_id") == b.get("machine_id") else "Different"
                ),
                (
                    "Failure Probability",
                    f'{a["failure_percentage"]:.2f}%',
                    f'{b["failure_percentage"]:.2f}%',
                    f'{delta:+.2f} pp'
                ),
                ("Risk Level", a["risk_level"], b["risk_level"], "—"),
                (
                    "Air Temperature (K)",
                    f'{a["air_temperature"]:.1f}',
                    f'{b["air_temperature"]:.1f}',
                    f'{b["air_temperature"] - a["air_temperature"]:+.1f}'
                ),
                (
                    "Process Temperature (K)",
                    f'{a["process_temperature"]:.1f}',
                    f'{b["process_temperature"]:.1f}',
                    f'{b["process_temperature"] - a["process_temperature"]:+.1f}'
                ),
                (
                    "Rotational Speed (RPM)",
                    f'{a["rotational_speed"]:.0f}',
                    f'{b["rotational_speed"]:.0f}',
                    f'{b["rotational_speed"] - a["rotational_speed"]:+.0f}'
                ),
                (
                    "Torque (Nm)",
                    f'{a["torque"]:.1f}',
                    f'{b["torque"]:.1f}',
                    f'{b["torque"] - a["torque"]:+.1f}'
                ),
                (
                    "Tool Wear (Minutes)",
                    f'{a["tool_wear"]:.0f}',
                    f'{b["tool_wear"]:.0f}',
                    f'{b["tool_wear"] - a["tool_wear"]:+.0f}'
                )
            ]
        )

        if delta > 0:
            st.warning(
                f"Prediction B is **{delta:.1f} percentage points riskier** "
                "than Prediction A."
            )

        elif delta < 0:
            st.success(
                f"Prediction B is **{abs(delta):.1f} percentage points safer** "
                "than Prediction A."
            )

        else:
            st.info("Both predictions have the same failure probability.")

    # MANAGE HISTORY

    st.markdown(
        '<div class="section-heading">Manage History</div>',
        unsafe_allow_html=True
    )

    confirm_clear = st.checkbox(
        "I understand this permanently deletes all of my prediction history.",
        key="history_confirm_clear"
    )

    if st.button(
        "Clear my history",
        disabled=not confirm_clear,
        key="history_clear_button"
    ):
        deleted = clear_predictions(user_id)
        st.session_state["flash"] = f"Deleted {deleted} prediction record(s)."
        st.rerun()


def render_profile_page(user_id):
    """Show the signed-in user's account details and usage summary."""
    user = st.session_state["user"]
    stats = get_prediction_stats(user_id)

    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">ACCOUNT CENTER</div>'
        '<div class="hero-title">User Profile</div>'
        '<div class="hero-description">'
        'View your account details and prediction activity.'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Account Details</div>',
        unsafe_allow_html=True
    )
    render_table(
        ["Field", "Information"],
        [
            ("Full name", user["name"]),
            ("Email address", user["email"]),
            ("Member since", user["created_at"]),
            ("Account storage", "Local MachineGuard database"),
            ("Password security", "PBKDF2-SHA256 protected"),
        ]
    )

    st.markdown(
        '<div class="section-heading">Prediction Activity</div>',
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", stats["total"])
    c2.metric("Failure Alerts", stats["failures"])
    c3.metric("Average Risk", f'{stats["avg_percentage"]:.1f}%')
    c4.metric("Highest Risk", f'{stats["max_percentage"]:.1f}%')
    if stats["last_prediction_at"]:
        st.caption(f'Last prediction: {stats["last_prediction_at"]}')


BATCH_REPORT_HEADINGS = {
    "machine_id": "Machine ID",
    "row_number": "Source Row",
    "status": "Status",
    "air_temperature": "Air Temperature (K)",
    "process_temperature": "Process Temperature (K)",
    "rotational_speed": "Rotational Speed (RPM)",
    "torque": "Torque (Nm)",
    "tool_wear": "Tool Wear (Minutes)",
    "failure_probability": "Failure Probability",
    "failure_percentage": "Failure Probability (%)",
    "prediction_code": "Prediction Code",
    "risk_level": "Risk Level",
    "maintenance_required": "Maintenance Required",
    "recommended_action": "Recommended Action",
    "failure_mode": "Likely Failure Mode",
    "failure_mode_reason": "Failure Mode Explanation",
    "error": "Validation Error",
}


def _batch_report_csv(report: pd.DataFrame, only_flagged: bool = False) -> bytes:
    """Serialize a batch report with human-readable column headings."""
    frame = report[report["maintenance_required"] == "YES"] if only_flagged else report
    frame = frame[list(BATCH_REPORT_HEADINGS)].rename(columns=BATCH_REPORT_HEADINGS)
    return frame.to_csv(index=False).encode("utf-8")


def render_batch_page(user_id):
    """Drag-and-drop CSV scoring with a downloadable maintenance report."""

    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">FLEET-WIDE ANALYSIS</div>'
        '<div class="hero-title">Batch Prediction</div>'
        '<div class="hero-description">'
        'Upload telemetry for an entire fleet at once. Every machine is scored '
        'and the report highlights the assets that need maintenance first.'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Upload Machine Data</div>'
        '<div class="section-description">'
        'Drag and drop a CSV with one machine per row. Five sensor columns '
        'are required; a machine ID column is optional.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.expander("Need a template? Download the expected CSV layout"):
        st.download_button(
            "Download CSV template",
            data=template_csv().encode("utf-8"),
            file_name="machineguard_batch_template.csv",
            mime="text/csv",
            key="batch_template_download"
        )
        st.caption(
            "Required headers: air_temperature, process_temperature, "
            "rotational_speed, torque, tool_wear. The AI4I names "
            "(e.g. \"Air temperature [K]\") are also accepted, and a "
            "machine_id, UDI or Equipment ID column is used as the asset label."
        )

    uploaded = st.file_uploader(
        "Upload machine telemetry (CSV)",
        type=["csv"],
        key="batch_upload",
        help=(
            f"Up to {MAX_BATCH_ROWS:,} rows. Required columns: air_temperature, "
            "process_temperature, rotational_speed, torque, tool_wear."
        )
    )

    if uploaded is None:
        st.info(
            "No file uploaded yet. Download the template above to see the "
            "expected format, then drag your CSV into the box."
        )
        return

    raw = uploaded.getvalue()

    if not raw.strip():
        st.error("The uploaded file is empty.")
        return

    digest = hashlib.sha256(raw).hexdigest()
    cache = st.session_state.get("batch_cache")

    # Score once per unique file. Downloads and reruns reuse the cached report.
    if cache is None or cache["digest"] != digest:

        try:
            try:
                frame = pd.read_csv(io.BytesIO(raw))
            except UnicodeDecodeError:
                frame = pd.read_csv(io.BytesIO(raw), encoding="latin-1")

        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.session_state.pop("batch_cache", None)
            return

        try:
            report, summary = build_batch_report(frame, predictor)

        except ValueError as exc:
            st.error(str(exc))
            st.session_state.pop("batch_cache", None)
            return

        cache = {
            "digest": digest,
            "file_name": uploaded.name,
            "report": report,
            "summary": summary,
        }
        st.session_state["batch_cache"] = cache
        st.session_state.pop("batch_save_result", None)

    report = cache["report"]
    summary = cache["summary"]
    file_name = cache["file_name"]

    st.success(
        f"Scored **{summary['total']:,} machine(s)** from `{file_name}`."
    )

    # SUMMARY METRICS

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machines Scored", f"{summary['total']:,}")
    c2.metric("Need Maintenance", f"{summary['flagged']:,}")
    c3.metric("Invalid Rows", f"{summary['invalid']:,}")
    c4.metric("Highest Risk", f"{summary['max_percentage']:.1f}%")

    if summary["valid"]:
        st.caption(
            f"{summary['flagged']} of {summary['valid']:,} valid machines "
            f"({summary['flagged_pct']:.1f}%) were flagged for maintenance. "
            f"Average failure probability: {summary['avg_percentage']:.1f}%."
        )

    if summary["top_machine"] is not None:
        st.warning(
            f"Highest-risk asset: **{summary['top_machine']}** at "
            f"{summary['max_percentage']:.1f}% failure probability. "
            "Review it first."
        )

    # RISK DISTRIBUTION

    st.markdown(
        '<div class="section-heading">Risk Distribution</div>',
        unsafe_allow_html=True
    )

    render_risk_distribution(summary["risk_counts"])

    # MAINTENANCE SHORTLIST

    flagged = report[report["maintenance_required"] == "YES"]

    st.markdown(
        f'<div class="section-heading">Maintenance Shortlist ({len(flagged)})</div>'
        '<div class="section-description">'
        'Machines flagged for inspection, highest failure probability first.'
        '</div>',
        unsafe_allow_html=True
    )

    if flagged.empty:
        st.success(
            "No machines were flagged for maintenance in this upload. "
            "All valid assets are within the model's safe operating range."
        )

    else:
        shortlist_limit = st.selectbox(
            "Rows to show",
            [10, 25, 50, 100, "All"],
            index=1,
            key="batch_shortlist_limit"
        )

        visible = (
            flagged if shortlist_limit == "All"
            else flagged.head(int(shortlist_limit))
        )

        render_table(
            ["Machine ID", "Source Row", "Failure Probability", "Risk Level",
             "Likely Failure Mode", "Recommended Action"],
            [
                (
                    row.machine_id,
                    row.row_number,
                    f"{row.failure_percentage:.2f}%",
                    row.risk_level,
                    row.failure_mode,
                    row.recommended_action,
                )
                for row in visible.itertuples()
            ]
        )

        if shortlist_limit != "All" and len(flagged) > int(shortlist_limit):
            st.caption(
                f"Showing the top {int(shortlist_limit)} of {len(flagged)} "
                "flagged machines. Download the CSV for the complete list."
            )

    # INVALID ROWS

    invalid = report[report["status"] != "OK"]

    if not invalid.empty:
        with st.expander(f"Rows that could not be scored ({len(invalid)})"):
            st.caption(
                "These rows were left out of the scoring and are marked "
                "REVIEW in the report. Fix the values and upload again."
            )
            render_table(
                ["Source Row", "Machine ID", "Validation Error"],
                [
                    (row.row_number, row.machine_id, row.error)
                    for row in invalid.head(50).itertuples()
                ]
            )
            if len(invalid) > 50:
                st.caption(f"Showing 50 of {len(invalid)} invalid rows.")

    # DOWNLOADS

    st.markdown(
        '<div class="section-heading">Download Reports</div>'
        '<div class="section-description">'
        'The full report includes every row; the maintenance report contains '
        'only the machines that need attention.'
        '</div>',
        unsafe_allow_html=True
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "Download full report (CSV)",
            data=_batch_report_csv(report),
            file_name=f"machineguard_batch_report_{stamp}.csv",
            mime="text/csv",
            use_container_width=True,
            key="batch_full_download"
        )

    with d2:
        st.download_button(
            "Download maintenance report (CSV)",
            data=_batch_report_csv(report, only_flagged=True),
            file_name=f"machineguard_maintenance_{stamp}.csv",
            mime="text/csv",
            use_container_width=True,
            key="batch_maintenance_download",
            disabled=flagged.empty
        )

    # SAVE TO HISTORY

    st.markdown(
        '<div class="section-heading">Save Results</div>'
        '<div class="section-description">'
        'Optionally add every valid machine from this upload to your personal '
        'prediction history.'
        '</div>',
        unsafe_allow_html=True
    )

    save_result = st.session_state.get("batch_save_result")

    if save_result and save_result["digest"] == digest:
        st.success(
            f"Saved {save_result['count']:,} prediction(s) to your history."
        )
        return

    if summary["valid"] == 0:
        st.caption("There are no valid rows to save.")
        return

    st.caption(
        f"This will add {summary['valid']:,} valid machine prediction(s) to "
        f"your history. Invalid rows are skipped."
    )

    if st.button(
        f"Save {summary['valid']:,} predictions to my history",
        key="batch_save_button"
    ):
        records = [
            {
                "machine_id": str(row.machine_id),
                "scenario": f"Batch upload · {row.machine_id}",
                "air_temperature": row.air_temperature,
                "process_temperature": row.process_temperature,
                "rotational_speed": row.rotational_speed,
                "torque": row.torque,
                "tool_wear": row.tool_wear,
                "failure_probability": row.failure_probability,
                "prediction_code": int(row.prediction_code),
                "risk_level": row.risk_level,
                "decision_threshold": float(predictor.threshold),
                "recommended_action": row.recommended_action,
                "failure_mode": row.failure_mode,
                "model_name": predictor.model_name,
            }
            for row in report[report["status"] == "OK"].itertuples()
        ]

        try:
            saved = save_predictions_bulk(user_id, records)

        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save the batch to your history: {exc}")

        else:
            st.session_state["batch_save_result"] = {
                "digest": digest,
                "count": saved,
            }
            st.rerun()


# Gate the application behind authentication.

session_expired = False
if "user" in st.session_state:
    expires_at = st.session_state.get("session_expires_at")
    if not expires_at or datetime.now().astimezone() >= datetime.fromisoformat(expires_at):
        st.session_state.clear()
        session_expired = True

if "user" not in st.session_state:
    if session_expired:
        st.warning("Your one-hour session has expired. Please sign in again.")
    render_auth_page()
    st.stop()

flash_message = st.session_state.pop("flash", None)

if flash_message:
    st.success(flash_message)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return EquipmentFailurePredictor()


try:
    predictor = load_model()

except Exception as e:
    st.error("Unable to load the trained machine learning model.")
    st.exception(e)
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    '<div class="project-brand">⚙️ Equipment Failure</div>'
    '<div class="project-subtitle">MACHINE LEARNING PROJECT</div>',
    unsafe_allow_html=True
)

current_user = st.session_state["user"]

st.sidebar.markdown(
    f'<div class="sidebar-user">'
    f'<div class="sidebar-user-name">{escape(current_user["name"])}</div>'
    f'<div class="sidebar-user-email">{escape(current_user["email"])}</div>'
    f'</div>',
    unsafe_allow_html=True
)

if st.sidebar.button(
    "Log out",
    use_container_width=True,
    key="logout_button"
):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("### NAVIGATION")

page = st.sidebar.radio(
    "Select Page",
    [
        "Prediction Dashboard",
        "Fleet Overview",
        "Batch Prediction",
        "Prediction History",
        "User Profile",
        "Model Information",
        "About Project"
    ],
    label_visibility="collapsed",
    key="nav_page"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### MODEL STATUS")

st.sidebar.success("Model Loaded")

st.sidebar.success("Random Forest Active")

st.sidebar.success("Prediction Engine Ready")

st.sidebar.markdown("---")

st.sidebar.markdown("### PROJECT TITLE")

st.sidebar.caption(
    "ML System for Early Detection of Equipment Failure"
)

st.sidebar.markdown("---")

st.sidebar.caption("Python | Streamlit | Scikit-learn")
expires_at = datetime.fromisoformat(st.session_state["session_expires_at"])
st.sidebar.caption(f"Session expires at {expires_at.strftime('%H:%M')}")


# =========================================================
# TOP HEADER
# =========================================================

st.markdown(
    '<div class="top-header">'
    '<div class="header-label">⚙️ MACHINE HEALTH MONITORING</div>'
    '<div class="header-status-group">'
    '<span class="header-status">● MODEL READY</span>'
    '<span class="header-status">● SESSION ACTIVE</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PREDICTION DASHBOARD
# =========================================================

if page == "Prediction Dashboard":

    # HERO SECTION

    hero1, hero2 = st.columns([4, 1])

    with hero1:

        st.markdown(
            '<div class="hero">'
            '<div class="hero-tag">MACHINE LEARNING BASED PREDICTIVE MAINTENANCE</div>'
            '<div class="hero-title">'
            'ML System for Early Detection of Equipment Failure'
            '</div>'
            '<div class="hero-description">'
            'Monitor machine health, analyze sensor parameters, and identify '
            'potential equipment failures using a trained machine learning model.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with hero2:

        st.markdown(
            '<div class="hero">'
            '<div class="hero-icon">⚙️</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # OVERVIEW

    st.markdown(
        '<div class="section-heading">System Overview</div>'
        '<div class="section-description">'
        'An overview of the machine learning prediction system'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    overview = [
        ("🧠", "ML ALGORITHM", "Random Forest", "Classification model"),
        ("📡", "SENSOR INPUTS", "5 Parameters", "Machine measurements"),
        ("⚙️", "DERIVED FEATURES", "3 Features", "Automatic calculation"),
        ("🟢", "MODEL STATUS", "ACTIVE", "Ready for prediction")
    ]

    for col, item in zip([c1, c2, c3, c4], overview):

        with col:

            st.markdown(
                f'<div class="overview-card">'
                f'<div class="overview-icon">{item[0]}</div>'
                f'<div class="overview-label">{item[1]}</div>'
                f'<div class="overview-value">{item[2]}</div>'
                f'<div class="overview-note">{item[3]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # SENSOR INPUT SECTION

    st.markdown(
        '<div class="section-heading">Machine Sensor Parameters</div>'
        '<div class="section-description">'
        'Enter the machine readings to analyze equipment failure risk'
        '</div>',
        unsafe_allow_html=True
    )

    scenarios = {
        "Custom Input": (298.2, 308.7, 1500.0, 40.0, 35.0),
        "Healthy Machine": (298.2, 308.7, 1500.0, 40.0, 35.0),
        "High Torque and Tool Wear": (301.5, 310.8, 1300.0, 65.5, 215.0),
        "Temperature Anomaly": (303.8, 311.2, 1320.0, 52.0, 140.0)
    }

    col_scen, col_mid = st.columns([1, 1])
    with col_scen:
        selected = st.selectbox(
            "Select Sample Machine Scenario",
            list(scenarios.keys())
        )
    with col_mid:
        single_machine_id = st.text_input(
            "Machine / Asset ID",
            value="MILL-01",
            placeholder="MILL-01",
            help="Unique asset identifier for equipment tracking (e.g. MILL-01, CNC-4)",
            key="single_machine_id"
        )

    defaults = scenarios[selected]

    with st.form("prediction_form"):

        # TEMPERATURE INPUTS

        st.markdown(
            '<div class="sensor-panel">'
            '<div class="sensor-title">🌡️ Temperature Sensors</div>'
            '<div class="sensor-description">'
            'Monitor air temperature and process temperature'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=float(defaults[0]),
                step=0.1
            )

        with col2:

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=float(defaults[1]),
                step=0.1
            )

        # MECHANICAL INPUTS

        st.markdown(
            '<div class="sensor-panel">'
            '<div class="sensor-title">⚙️ Mechanical Parameters</div>'
            '<div class="sensor-description">'
            'Monitor rotational speed, torque and tool wear'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        col3, col4, col5 = st.columns(3)

        with col3:

            rotational_speed = st.number_input(
                "Rotational Speed (RPM)",
                min_value=0.0,
                max_value=10000.0,
                value=float(defaults[2]),
                step=10.0
            )

        with col4:

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=float(defaults[3]),
                step=0.5
            )

        with col5:

            tool_wear = st.number_input(
                "Tool Wear (Minutes)",
                min_value=0.0,
                max_value=1000.0,
                value=float(defaults[4]),
                step=1.0
            )

        st.info(
            "Three additional physics-based features are calculated "
            "automatically by the prediction model."
        )

        submitted = st.form_submit_button(
            "🔍 PREDICT EQUIPMENT FAILURE",
            use_container_width=True
        )

    # PREDICTION

    if submitted:

        if air_temp <= 0 or process_temp <= 0:

            st.error("Please enter valid temperature readings.")

        else:

            try:

                with st.spinner("Analyzing machine sensor readings..."):

                    result = predictor.predict(
                        air_temperature=air_temp,
                        process_temperature=process_temp,
                        rotational_speed=rotational_speed,
                        torque=torque,
                        tool_wear=tool_wear
                    )

                normalized_mid = normalize_machine_id(single_machine_id)
                try:
                    save_prediction(
                        user_id=st.session_state["user"]["id"],
                        scenario=selected,
                        air_temperature=air_temp,
                        process_temperature=process_temp,
                        rotational_speed=rotational_speed,
                        torque=torque,
                        tool_wear=tool_wear,
                        result=result,
                        machine_id=normalized_mid,
                    )
                    saved_to_history = True
                    save_error_message = ""

                except Exception as save_error:
                    saved_to_history = False
                    save_error_message = str(save_error)

                probability = result["failure_probability"] * 100

                risk = result["risk_level"]

                is_failure = result["prediction_code"] == 1

                st.markdown("---")

                st.markdown(
                    '<div class="section-heading">Prediction Results</div>'
                    '<div class="section-description">'
                    'Machine health analysis based on the submitted sensor values'
                    '</div>',
                    unsafe_allow_html=True
                )

                # RESULT BANNER

                if is_failure:

                    st.markdown(
                        f'<div class="failure-result">'
                        f'<div class="result-title">⚠️ FAILURE PREDICTED</div>'
                        f'<div class="result-description">'
                        f'Asset ID: <b>{escape(normalized_mid)}</b> &nbsp;|&nbsp; '
                        f'Risk Level: <b>{risk}</b><br>'
                        f'Potential equipment failure pattern detected. '
                        f'Please inspect the machine.'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f'<div class="success-result">'
                        f'<div class="result-title">✅ NORMAL OPERATION</div>'
                        f'<div class="result-description">'
                        f'Asset ID: <b>{escape(normalized_mid)}</b> &nbsp;|&nbsp; '
                        f'Risk Level: <b>{risk}</b><br>'
                        f'The machine is classified as normal by the prediction model.'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                # RESULT METRICS

                r0, r1, r2, r3 = st.columns(4)

                with r0:

                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="result-label">MACHINE / ASSET ID</div>'
                        f'<div class="result-value">{escape(normalized_mid)}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with r1:

                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="result-label">FAILURE PROBABILITY</div>'
                        f'<div class="result-value">{probability:.2f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with r2:

                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="result-label">RISK LEVEL</div>'
                        f'<div class="result-value">{risk}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with r3:

                    threshold = result["decision_threshold"] * 100

                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="result-label">DECISION THRESHOLD</div>'
                        f'<div class="result-value">{threshold:.0f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # PROBABILITY

                st.markdown(
                    '<div class="section-heading">Failure Probability Meter</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    min(max(probability / 100, 0.0), 1.0)
                )

                st.caption(
                    "The model predicts failure when its estimated probability "
                    "reaches or exceeds the configured decision threshold."
                )

                # MAINTENANCE ACTION

                st.markdown(
                    '<div class="section-heading">'
                    '🛠️ Recommended Maintenance Action'
                    '</div>',
                    unsafe_allow_html=True
                )

                if risk in ["HIGH", "CRITICAL"]:

                    st.error(result["recommended_action"])

                elif risk == "MODERATE":

                    st.warning(result["recommended_action"])

                else:

                    st.success(result["recommended_action"])

                st.markdown(
                    '<div class="section-heading">Failure Mode Breakdown</div>'
                    '<div class="section-description">'
                    'A physics-based explanation of the telemetry pattern behind '
                    'the binary model result'
                    '</div>',
                    unsafe_allow_html=True
                )
                mode = result["failure_mode"]
                if is_failure and mode["mode_code"] != "NONE":
                    st.warning(
                        f'Likely mode: **{mode["mode_label"]}** — {mode["reason"]}'
                    )
                else:
                    st.info(
                        f'Likely mode: **{mode["mode_label"]}** — {mode["reason"]}'
                    )
                st.caption(
                    "This is a physics-based indicator derived from known AI4I "
                    "failure thresholds; the trained Random Forest itself predicts "
                    "overall failure probability, not failure-mode classes."
                )

                # SENSOR SUMMARY

                st.markdown(
                    '<div class="section-heading">Sensor Reading Summary</div>',
                    unsafe_allow_html=True
                )

                render_table(
                    ["Sensor Parameter", "Reading"],
                    [
                        (key, value)
                        for key, value in result["telemetry_inputs"].items()
                    ]
                )

                if saved_to_history:
                    st.caption(
                        "✅ This prediction was saved to your history. "
                        "Open **Prediction History** in the sidebar to review it."
                    )

                elif save_error_message:
                    st.warning(
                        "The prediction was computed but could not be saved "
                        f"to your history: {save_error_message}"
                    )

                # TECHNICAL DETAILS

                with st.expander("View Technical Model Details"):

                    st.markdown("### Model Information")

                    st.json(result["model_provenance"])

                    st.caption(result["disclaimer"])

            except Exception as e:

                st.error("Prediction failed. Please check the input values.")

                st.exception(e)


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    render_batch_page(st.session_state["user"]["id"])


# =========================================================
# FLEET OVERVIEW DASHBOARD
# =========================================================

elif page == "Fleet Overview":

    render_fleet_page(st.session_state["user"]["id"])


# =========================================================
# PREDICTION HISTORY
# =========================================================

elif page == "Prediction History":

    render_history_page(st.session_state["user"]["id"])


# =========================================================
# USER PROFILE
# =========================================================

elif page == "User Profile":

    render_profile_page(st.session_state["user"]["id"])


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">MACHINE LEARNING DETAILS</div>'
        '<div class="hero-title">Model Information</div>'
        '<div class="hero-description">'
        'Technical information, input features and evaluation performance '
        'of the trained machine learning model.'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Machine Learning Algorithm</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Optimized Random Forest Classifier with Physics-Based Feature Engineering"
    )

    # FEATURE CARDS

    st.markdown(
        '<div class="section-heading">Model Features</div>',
        unsafe_allow_html=True
    )

    f1, f2 = st.columns(2)

    with f1:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-card-title">📡 Raw Sensor Features</div>'
            '<div class="info-card-description">'
            '1. Air Temperature (K)<br>'
            '2. Process Temperature (K)<br>'
            '3. Rotational Speed (RPM)<br>'
            '4. Torque (Nm)<br>'
            '5. Tool Wear (Minutes)'
            '</div></div>',
            unsafe_allow_html=True
        )

    with f2:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-card-title">⚙️ Derived Features</div>'
            '<div class="info-card-description">'
            '1. Temperature Difference<br>'
            '2. Mechanical Power<br>'
            '3. Overstrain Index'
            '</div></div>',
            unsafe_allow_html=True
        )

    # PERFORMANCE

    st.markdown(
        '<div class="section-heading">Model Test Performance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "98.95%")
    c2.metric("Precision", "85.07%")
    c3.metric("Recall", "83.82%")
    c4.metric("F1 Score", "84.44%")

    # CONFUSION MATRIX

    st.markdown(
        '<div class="section-heading">Confusion Matrix Results</div>',
        unsafe_allow_html=True
    )

    render_table(
        ["Prediction Result", "Count"],
        [
            ("Correctly Detected Failures", 57),
            ("False Alarms", 10),
            ("Missed Failures", 11),
            ("Correct Normal Predictions", 1922)
        ]
    )

    st.caption(
        "These metrics are retained from the supplied model evaluation report. "
        "They represent test dataset performance and do not guarantee real-world accuracy."
    )

    st.warning(
        "Accuracy alone is not enough to evaluate equipment failure detection. "
        "Recall and missed failures are also important."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">PROJECT OVERVIEW</div>'
        '<div class="hero-title">'
        'ML System for Early Detection of Equipment Failure'
        '</div>'
        '<div class="hero-description">'
        'A machine learning based predictive maintenance system '
        'that identifies potential equipment failures using sensor data.'
        '</div></div>',
        unsafe_allow_html=True
    )

    # OBJECTIVE

    st.markdown(
        '<div class="section-heading">Project Objective</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The main objective of this project is to develop a machine learning
        system that predicts possible equipment failure using machine sensor
        readings.

        The system analyzes temperature, rotational speed, torque and tool wear
        to identify patterns associated with equipment failure.

        By using a trained machine learning model, the system estimates
        failure probability and provides maintenance recommendations.
        """
    )

    # TECHNOLOGIES

    st.markdown(
        '<div class="section-heading">Technologies Used</div>',
        unsafe_allow_html=True
    )

    render_table(
        ["Technology", "Purpose"],
        [
            ("Python", "Programming Language"),
            ("Streamlit", "Frontend Dashboard"),
            ("Scikit-learn", "Machine Learning Framework"),
            ("Random Forest", "Prediction Algorithm"),
            ("Pandas", "Data Processing"),
            ("NumPy", "Numerical Computation"),
            ("Joblib", "Model Serialization")
        ]
    )

    # WORKFLOW

    st.markdown(
        '<div class="section-heading">How the System Works</div>',
        unsafe_allow_html=True
    )

    workflow = [
        (
            "01",
            "Sensor Data Collection",
            "Five machine sensor measurements are provided as input."
        ),
        (
            "02",
            "Data Processing",
            "The sensor readings are processed by the application."
        ),
        (
            "03",
            "Feature Engineering",
            "Three additional physics-based features are calculated."
        ),
        (
            "04",
            "Machine Learning",
            "The trained Random Forest model analyzes the readings."
        ),
        (
            "05",
            "Failure Prediction",
            "The model generates a failure probability and risk category."
        ),
        (
            "06",
            "Maintenance Recommendation",
            "The system displays a recommended maintenance action."
        )
    ]

    for number, title, description in workflow:

        st.markdown(
            f'<div class="workflow-card">'
            f'<div class="workflow-number">{number}</div>'
            f'<div class="workflow-title">{title}</div>'
            f'<div class="workflow-description">{description}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # DATASET

    st.markdown(
        '<div class="section-heading">Dataset Information</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The project uses the AI4I 2020 Predictive Maintenance Dataset "
        "for training and evaluating the machine learning model."
    )

    st.warning(
        "This system estimates statistical failure risk based on dataset patterns. "
        "It does not guarantee the exact time of equipment breakdown. "
        "Real-world maintenance decisions require qualified inspection."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '<div class="footer-title">'
    'ML System for Early Detection of Equipment Failure'
    '</div>'
    '<br>'
    'Powered by Python | Streamlit | Scikit-learn'
    '<br><br>'
    'Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
