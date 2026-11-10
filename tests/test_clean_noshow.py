import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from batch_risk_control.clean_noshow import build_batches, FEATURE_COLS


def _toy_df():
    return pd.DataFrame({
        "PatientId": [1, 2, 3, 4, 5],
        "Neighbourhood": ["A", "A", "A", "B", "B"],
        "AppointmentDay": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-01"] * 2),
        "batch_key": ["A||2020-01-01"] * 3 + ["B||2020-01-01"] * 2,
        "day_of_week": ["Wednesday"] * 5,
        "Attended": [1, 0, 1, 1, 0],
        "Age": [30, 40, 50, 20, 60],
        "Gender_M": [0, 1, 0, 1, 1],
        "Scholarship": [0, 0, 1, 0, 0],
        "Hipertension": [0, 1, 0, 0, 1],
        "Diabetes": [0, 0, 0, 1, 0],
        "Alcoholism": [0, 0, 0, 0, 0],
        "Handcap": [0, 0, 0, 0, 1],
        "SMS_received": [1, 0, 1, 0, 1],
        "lead_days": [5, 10, 0, 3, 20],
    })


def test_build_batches_groups_by_neighbourhood_and_date():
    df = _toy_df()
    batches = build_batches(df)
    assert len(batches) == 2
    sizes = sorted(len(b["Y"]) for b in batches)
    assert sizes == [2, 3]


def test_build_batches_feature_matrix_shape_matches_feature_cols():
    df = _toy_df()
    batches = build_batches(df)
    for b in batches:
        assert b["X"].shape == (len(b["Y"]), len(FEATURE_COLS))


def test_build_batches_Y_matches_attended_flag():
    df = _toy_df()
    batches = build_batches(df)
    a = [b for b in batches if b["neighbourhood"] == "A"][0]
    np.testing.assert_array_equal(sorted(a["Y"]), [0, 1, 1])


def test_build_batches_patient_ids_preserved():
    df = _toy_df()
    batches = build_batches(df)
    all_ids = np.concatenate([b["patient_ids"] for b in batches])
    assert sorted(all_ids) == [1, 2, 3, 4, 5]
