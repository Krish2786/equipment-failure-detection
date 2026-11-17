
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
    get_prediction_stats,
    get_predictions,
    init_db,
    save_prediction,
    save_predictions_bulk,
)


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
        'Narrow the history by risk level, outcome, or number of rows.'
        '</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        risk_filter = st.selectbox(
            "Risk level",
            ["All", "LOW", "MODERATE", "HIGH", "CRITICAL"],
            key="history_risk_filter"
        )

    with f2:
        outcome_filter = st.selectbox(
            "Outcome",
            ["All", "Failure", "Normal"],
            key="history_outcome_filter"
        )

    with f3:
        limit_choice = st.selectbox(
            "Rows to show",
            [10, 25, 50, 100, "All"],
            index=1,
            key="history_limit"
        )

    filtered = predictions

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
            "Scenario",
            "Failure Probability",
            "Risk Level",
            "Outcome"
        ],
        [
            [
                p["id"],
                p["created_at"],
                p["scenario"] or "Custom Input",
                f'{p["failure_percentage"]:.1f}%',
                p["risk_level"],
                "Failure" if p["prediction_code"] == 1 else "Normal"
            ]
            for p in filtered
        ]
    )

    # CSV EXPORT

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow([
        "ID", "Date & Time", "Scenario",
        "Air Temperature (K)", "Process Temperature (K)",
        "Rotational Speed (RPM)", "Torque (Nm)", "Tool Wear (Minutes)",
        "Failure Probability (%)", "Risk Level", "Prediction Code",
        "Decision Threshold", "Recommended Action", "Model"
    ])

    for p in predictions:
        writer.writerow([
            p["id"],
            p["created_at"],
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
        f'#{p["id"]} · {p["created_at"]} · {p["failure_percentage"]:.1f}% · {p["risk_level"]}': p
        for p in predictions
    }

    selected_label = st.selectbox(
        "Select a record",
        list(records_by_label.keys()),
        key="history_detail_select"
    )

    record = records_by_label[selected_label]

    d1, d2, d3 = st.columns(3)
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

    selected = st.selectbox(
        "Select Sample Machine Scenario",
        list(scenarios.keys())
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

                try:
                    save_prediction(
                        user_id=st.session_state["user"]["id"],
                        scenario=selected,
                        air_temperature=air_temp,
                        process_temperature=process_temp,
                        rotational_speed=rotational_speed,
                        torque=torque,
                        tool_wear=tool_wear,
                        result=result
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
                        f'Risk Level: <b>{risk}</b><br>'
                        f'The machine is classified as normal by the prediction model.'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                # RESULT METRICS

                r1, r2, r3 = st.columns(3)

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
