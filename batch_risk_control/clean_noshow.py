"""Clean the raw Kaggle Medical Appointment No-Shows CSV into a batch-ready
panel. Batch key verified in literature review: (Neighbourhood x
AppointmentDay) -- Neighbourhood is the CLINIC's location per the dataset's
own documentation, not the patient's home, so this is a genuine shared-
capacity session, not an arbitrary grouping.
"""
import numpy as np
import pandas as pd


def load_and_clean(path="data/raw/KaggleV2-May-2016.csv"):
    df = pd.read_csv(path)
    assert len(df) == 110527, f"expected 110527 rows, got {len(df)}"

    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["No_show"] = (df["No-show"] == "Yes").astype(int)
    df["Attended"] = 1 - df["No_show"]

    # documented data-quality issue: one row has Age == -1
    before = len(df)
    df = df[df["Age"] >= 0].copy()
    dropped_age = before - len(df)

    # lead time between scheduling and the appointment itself (days);
    # a handful of rows have negative lead time (data-entry timestamp
    # errors, documented in community analyses of this dataset) -- drop
    before2 = len(df)
    df["lead_days"] = (df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()).dt.days
    df = df[df["lead_days"] >= 0].copy()
    dropped_lead = before2 - len(df)

    df["day_of_week"] = df["AppointmentDay"].dt.day_name()
    df["Gender_M"] = (df["Gender"] == "M").astype(int)
    df["batch_key"] = df["Neighbourhood"] + "||" + df["AppointmentDay"].dt.strftime("%Y-%m-%d")

    print(f"dropped {dropped_age} rows with invalid Age, {dropped_lead} rows with negative lead_days")
    print(f"final: {len(df)} rows, {df['batch_key'].nunique()} batches, "
          f"{df['PatientId'].nunique()} unique patients")
    return df


FEATURE_COLS = ["Age", "Gender_M", "Scholarship", "Hipertension", "Diabetes",
                 "Alcoholism", "Handcap", "SMS_received", "lead_days"]


def build_batches(df):
    """Return a list of dicts, one per (Neighbourhood, AppointmentDay) batch,
    matching the same schema as the simulation DGP's output (X, Y, plus
    real metadata) so the SAME methods.py functions apply unchanged.
    """
    batches = []
    for key, g in df.groupby("batch_key"):
        X = g[FEATURE_COLS].to_numpy(dtype=float)
        Y = g["Attended"].to_numpy(dtype=int)
        batches.append({
            "X": X, "Y": Y, "batch_key": key,
            "neighbourhood": g["Neighbourhood"].iloc[0],
            "appointment_day": g["AppointmentDay"].iloc[0],
            "day_of_week": g["day_of_week"].iloc[0],
            "patient_ids": g["PatientId"].to_numpy(),
        })
    return batches


if __name__ == "__main__":
    df = load_and_clean()
    batches = build_batches(df)
    sizes = np.array([len(b["Y"]) for b in batches])
    print(f"\n{len(batches)} batches, size min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.2f} median={np.median(sizes):.0f}")
    print(f"date range: {df['AppointmentDay'].min()} to {df['AppointmentDay'].max()} "
          f"({df['AppointmentDay'].dt.date.nunique()} distinct days)")
    print(f"overall attendance rate: {df['Attended'].mean():.4f}")
