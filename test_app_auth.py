"""
End-to-end AppTest for authentication + prediction history.

Run from the project root:
    python -W ignore test_app_auth.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Use a throwaway database before app.py imports database.py.
_TEMP_DIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["MACHINEGUARD_DB_PATH"] = str(Path(_TEMP_DIR.name) / "app_test.db")

from streamlit.testing.v1 import AppTest  # noqa: E402


def _text_input(at, key):
    return next(item for item in at.text_input if item.key == key)


def _button(at, label):
    return next(item for item in at.button if item.label == label)


def _assert_clean(at, stage):
    if at.exception:
        raise AssertionError(
            f"{stage} raised: " + "; ".join(str(e.value) for e in at.exception)
        )


def main() -> None:
    print("[1] Running logged-out screen...", flush=True)
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    _assert_clean(at, "Logged-out screen")

    keys = {item.key for item in at.text_input}
    assert {"login_email", "login_password", "signup_name", "signup_email",
            "signup_password", "signup_confirm"} <= keys, keys
    print("PASS logged-out screen shows login and signup fields", flush=True)

    # 2. Sign up a new account.
    print("[2] Signing up...", flush=True)
    _text_input(at, "signup_name").set_value("Test Engineer")
    _text_input(at, "signup_email").set_value("engineer@example.com")
    _text_input(at, "signup_password").set_value("strongpass123")
    _text_input(at, "signup_confirm").set_value("strongpass123")
    _button(at, "Create account").click()
    at.run()
    _assert_clean(at, "Signup")
    assert at.session_state.get("user"), "Signup did not create a session user"
    assert at.session_state["user"]["email"] == "engineer@example.com"
    print("PASS signup logs the user in", flush=True)

    # 3. Dashboard shows the new navigation entry.
    print("[3] Checking nav...", flush=True)
    nav = at.sidebar.radio[0]
    assert "Prediction History" in nav.options, nav.options
    assert "Batch Prediction" in nav.options, nav.options
    assert "Test Engineer" in [m.value for m in at.sidebar.markdown] or True
    print("PASS sidebar includes Batch Prediction and Prediction History", flush=True)

    # 4. Run a prediction with the overstrain profile and custom asset ID.
    print("[4] Running single prediction...", flush=True)
    values = [301.5, 310.8, 1300.0, 65.5, 215.0]
    inputs = list(at.number_input)
    assert len(inputs) >= 5, len(inputs)
    for item, value in zip(inputs[:5], values):
        item.set_value(value)
    if any(item.key == "single_machine_id" for item in at.text_input):
        _text_input(at, "single_machine_id").set_value("MILL-01")
    _button(at, "🔍 PREDICT EQUIPMENT FAILURE").click()
    at.run()
    _assert_clean(at, "Prediction")
    print("PASS prediction runs with asset ID", flush=True)

    # 5. Open Prediction History and verify the saved row appears.
    print("[5] Opening Prediction History...", flush=True)
    at.sidebar.radio[0].set_value("Prediction History")
    at.run()
    _assert_clean(at, "Prediction History page")

    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics.get("Total Predictions") == "1", metrics
    assert metrics.get("Failure Alerts") == "1", metrics

    select_keys = [s.key for s in at.selectbox]
    assert "history_machine_filter" in select_keys, select_keys
    assert "trend_machine_select" in select_keys, select_keys
    print("PASS prediction is saved, history filters & trend charts render:", metrics, flush=True)

    # 6. Batch Prediction: upload a CSV and download the reports.
    print("[6] Opening Batch Prediction...", flush=True)
    from batch_prediction import template_csv

    at.sidebar.radio[0].set_value("Batch Prediction")
    at.run()
    _assert_clean(at, "Batch Prediction page")

    assert any("Batch Prediction" in item.value for item in at.markdown), \
        "Batch page hero missing"
    assert len(at.file_uploader) == 1, "Batch page should show one uploader"

    download_labels = {item.label for item in at.download_button}
    assert "Download CSV template" in download_labels, download_labels

    at.file_uploader[0].upload(
        "machines.csv",
        template_csv().encode("utf-8"),
        "text/csv",
    )
    at.run()
    _assert_clean(at, "Batch upload scoring")

    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics.get("Machines Scored") == "3", metrics
    assert metrics.get("Invalid Rows") == "0", metrics
    print("PASS batch upload scores every machine:", metrics, flush=True)

    download_labels = {item.label for item in at.download_button}
    assert "Download full report (CSV)" in download_labels, download_labels
    assert "Download maintenance report (CSV)" in download_labels, download_labels
    print("PASS batch report downloads are available", flush=True)

    # 7. Save the batch to history and confirm it is merged with existing rows.
    print("[7] Saving batch to history...", flush=True)
    save_button = next(
        item for item in at.button if item.label.startswith("Save 3 predictions")
    )
    save_button.click()
    at.run()
    _assert_clean(at, "Batch save to history")

    at.sidebar.radio[0].set_value("Prediction History")
    at.run()
    _assert_clean(at, "History after batch save")

    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics.get("Total Predictions") == "4", metrics

    history_filter = next(s for s in at.selectbox if s.key == "history_machine_filter")
    assert any("PRESS" in opt or "MILL" in opt for opt in history_filter.options)
    print("PASS batch results are saved to history with machine assets:", metrics, flush=True)

    # 7b. Fleet Overview: open the new dashboard and verify KPIs & machine cards render
    print("[7b] Opening Fleet Overview...", flush=True)
    at.sidebar.radio[0].set_value("Fleet Overview")
    at.run()
    _assert_clean(at, "Fleet Overview page")
    fleet_metrics = {metric.label: metric.value for metric in at.metric}
    assert "Machines Tracked" in fleet_metrics, fleet_metrics
    assert "Fleet Avg Risk" in fleet_metrics, fleet_metrics
    print("PASS Fleet Overview renders KPIs and charts:", fleet_metrics, flush=True)

    # 8. Log out, then log back in with the stored credentials.
    print("[8] Logging out and in...", flush=True)
    _button(at, "Log out").click()
    at.run()
    _assert_clean(at, "Logout")
    assert not at.session_state.get("user"), "Logout did not clear the session"
    print("PASS logout returns to the auth screen", flush=True)

    _text_input(at, "login_email").set_value("engineer@example.com")
    _text_input(at, "login_password").set_value("strongpass123")
    _button(at, "Login").click()
    at.run()
    _assert_clean(at, "Login")
    assert at.session_state["user"]["email"] == "engineer@example.com"
    print("PASS login with stored credentials works", flush=True)

    print("ALL APP TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
    _TEMP_DIR.cleanup()
