"""Clean the Hotel Booking Demand dataset into a batch-ready panel.

IMPORTANT framing note (a real trap caught in the literature review, not
hypothetical): this dataset's `is_canceled` field (37.04% rate) conflates
advance cancellations (usually recoverable/resellable capacity) with true
last-minute no-shows (`reservation_status == 'No-Show'`, only 1.01%, too
rare for a statistically rich second validation domain). We use
`is_canceled` here, but explicitly reframed as CAPACITY-RELEASE RISK --
will this booking fail to consume its reserved capacity, for any reason
(cancellation or no-show)? This is a real, distinct, still OR-relevant
overbooking-adjacent problem (a hotel deciding how many rooms to release
for resale given combined cancellation+no-show risk), not a silent
relabeling of "no-show". Y=1 means the booking IS fulfilled (checks out),
matching the medical dataset's Y=1=Attended convention.

Batch key verified in literature review: (hotel x arrival_date) -- two
hotels, arrivals spanning 2015-07 to 2017-08, giving 1,586 batches.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = ["lead_time", "is_repeated_guest", "previous_cancellations",
                 "previous_bookings_not_canceled", "total_of_special_requests",
                 "required_car_parking_spaces", "adr", "non_refundable_deposit"]


def load_and_clean(path="data/raw/hotel_bookings.csv"):
    df = pd.read_csv(path)
    assert len(df) == 119390, f"expected 119390 rows, got {len(df)}"

    df["Fulfilled"] = 1 - df["is_canceled"]
    df["non_refundable_deposit"] = (df["deposit_type"] == "Non Refund").astype(int)

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_date_month"] + "-" +
        df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    df["batch_key"] = df["hotel"] + "||" + df["arrival_date"].dt.strftime("%Y-%m-%d")

    before = len(df)
    df = df.dropna(subset=["adr"]).copy()
    df = df[df["adr"] >= 0].copy()  # a documented data artifact: a few negative ADR rows
    dropped = before - len(df)

    print(f"dropped {dropped} rows with missing/invalid adr")
    print(f"final: {len(df)} rows, {df['batch_key'].nunique()} batches, "
          f"fulfillment rate = {df['Fulfilled'].mean():.4f}")
    return df


def build_batches(df):
    batches = []
    for key, g in df.groupby("batch_key"):
        X = g[FEATURE_COLS].to_numpy(dtype=float)
        Y = g["Fulfilled"].to_numpy(dtype=int)
        batches.append({
            "X": X, "Y": Y, "batch_key": key,
            "hotel": g["hotel"].iloc[0],
            "arrival_date": g["arrival_date"].iloc[0],
        })
    return batches


if __name__ == "__main__":
    df = load_and_clean()
    batches = build_batches(df)
    sizes = np.array([len(b["Y"]) for b in batches])
    print(f"\n{len(batches)} batches, size min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.2f} median={np.median(sizes):.0f}")
    print(f"date range: {df['arrival_date'].min()} to {df['arrival_date'].max()} "
          f"({df['arrival_date'].dt.date.nunique()} distinct days)")
