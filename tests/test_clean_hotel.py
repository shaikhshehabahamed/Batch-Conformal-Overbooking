import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from batch_risk_control.clean_hotel import build_batches, FEATURE_COLS


def _toy_df():
    return pd.DataFrame({
        "hotel": ["Resort Hotel"] * 3 + ["City Hotel"] * 2,
        "arrival_date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-01"] * 2),
        "batch_key": ["Resort Hotel||2020-01-01"] * 3 + ["City Hotel||2020-01-01"] * 2,
        "Fulfilled": [1, 0, 1, 1, 1],
        "lead_time": [10, 20, 30, 5, 8],
        "is_repeated_guest": [0, 0, 1, 0, 0],
        "previous_cancellations": [0, 1, 0, 0, 0],
        "previous_bookings_not_canceled": [0, 0, 2, 0, 1],
        "total_of_special_requests": [1, 0, 2, 0, 1],
        "required_car_parking_spaces": [0, 0, 1, 0, 0],
        "adr": [100.0, 80.0, 120.0, 90.0, 95.0],
        "non_refundable_deposit": [0, 1, 0, 0, 0],
    })


def test_build_batches_groups_by_hotel_and_date():
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


def test_build_batches_Y_matches_fulfilled_flag():
    df = _toy_df()
    batches = build_batches(df)
    resort = [b for b in batches if b["hotel"] == "Resort Hotel"][0]
    np.testing.assert_array_equal(sorted(resort["Y"]), [0, 1, 1])
