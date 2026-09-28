"""Raw data acquisition, validation, and SQL preprocessing (DuckDB)."""
from __future__ import annotations

import urllib.request
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd
import pyreadr
from sklearn.model_selection import train_test_split

from causalads import config as C

FEATURES: list[str] = [
    "recency", "history", "log_history", "history_bucket", "history_at_floor", "mens", "womens",
    "both_lines", "newbie", "recent_buyer", "zip_urban", "zip_suburban",
    "zip_rural", "ch_phone", "ch_web", "ch_multi",
]
OUTCOMES: list[str] = ["visit", "conversion", "spend"]

RAW_COLUMNS = {
    "recency", "history_segment", "history", "mens", "womens", "zip_code",
    "newbie", "channel", "segment", "visit", "conversion", "spend",
}


def download() -> None:
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    targets = {"hillstrom.csv": C.HILLSTROM_URL, **C.GEOLIFT_URLS}
    for name, url in targets.items():
        path = C.DATA_RAW / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)


def validate_raw(raw: pd.DataFrame) -> dict:
    """Schema/range checks. Raises on anything that would silently corrupt results."""
    missing_cols = RAW_COLUMNS - set(raw.columns)
    if missing_cols:
        raise ValueError(f"raw extract missing columns: {sorted(missing_cols)}")
    checks = {
        "rows": int(len(raw)),
        "missing_values": {c: int(raw[c].isna().sum()) for c in sorted(RAW_COLUMNS)},
        "duplicate_rows": int(raw.duplicated().sum()),
        "recency_in_1_12": bool(raw["recency"].between(1, 12).all()),
        "history_positive": bool((raw["history"] > 0).all()),
        "binary_flags_ok": bool(all(raw[c].isin([0, 1]).all() for c in ["mens", "womens", "newbie", "visit", "conversion"])),
        "spend_nonneg": bool((raw["spend"] >= 0).all()),
        "spend_only_if_converted": bool(((raw["spend"] > 0) <= (raw["conversion"] == 1)).all()),
        "segments": sorted(raw["segment"].unique().tolist()),
        "zip_code_misspelled_surburban": int((raw["zip_code"] == "Surburban").sum()),
        "history_at_floor_29_99": int((raw["history"] <= 29.99).sum()),
        "note": "duplicate_rows are distinct customers sharing the $29.99 history floor "
                "and identical categorical features; they are kept.",
    }
    if not (checks["recency_in_1_12"] and checks["binary_flags_ok"] and checks["spend_nonneg"]):
        raise ValueError(f"raw extract failed validation: {checks}")
    return checks


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value policy. The 2008 extract has none, but the pipeline must not
    assume that: numeric -> median, flags/categoricals -> mode, plus a
    `<col>_missing` indicator whenever anything was filled."""
    df = df.copy()
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0 or col in OUTCOMES:
            continue
        df[f"{col}_missing"] = df[col].isna().astype(int)
        fill = df[col].median() if pd.api.types.is_float_dtype(df[col]) else df[col].mode().iloc[0]
        df[col] = df[col].fillna(fill)
    return df


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    """Return one row per customer: id, features, arm/treatment, outcomes."""
    download()
    raw = pd.read_csv(C.DATA_RAW / "hillstrom.csv")
    raw.insert(0, "customer_id", np.arange(len(raw)))
    raw = _impute(raw)
    con = duckdb.connect()
    con.register("raw", raw)
    for script in ["01_clean.sql", "02_features.sql"]:
        con.execute((C.SQL_DIR / script).read_text())
    df = con.execute(
        """
        SELECT f.*, c.zip_code, c.channel, c.history_segment, c.segment,
               c.arm, c.treated, c.visit, c.conversion, c.spend
        FROM features f JOIN customers c USING (customer_id)
        ORDER BY customer_id
        """
    ).df()
    con.close()
    assert len(df) == len(raw) and df["arm"].notna().all()
    return df


def arm_summary() -> pd.DataFrame:
    df = load()
    con = duckdb.connect()
    con.register("customers", df)
    out = con.execute((C.SQL_DIR / "03_arm_summary.sql").read_text()).df()
    con.close()
    return out


@lru_cache(maxsize=1)
def splits() -> pd.Series:
    """Fixed train/val/test assignment (60/20/20), stratified on arm x visit x conversion.
    Every stage uses this so no stage ever trains on another stage's test rows."""
    path = C.DATA_PROCESSED / "splits.csv"
    df = load()
    if path.exists():
        s = pd.read_csv(path, index_col="customer_id")["split"]
        if len(s) == len(df):
            return s
    strata = df["arm"].astype(str) + df["visit"].astype(str) + df["conversion"].astype(str)
    ids = df["customer_id"].to_numpy()
    trval, test = train_test_split(ids, test_size=C.TEST_FRAC, stratify=strata, random_state=C.SEED)
    val_share = C.VAL_FRAC / (1 - C.TEST_FRAC)
    train, val = train_test_split(trval, test_size=val_share, stratify=strata.iloc[trval], random_state=C.SEED)
    s = pd.Series("train", index=pd.Index(ids, name="customer_id"), name="split")
    s.loc[val] = "val"
    s.loc[test] = "test"
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(path)
    return s


def split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load()
    s = splits().reindex(df["customer_id"]).to_numpy()
    return (df[s == "train"].reset_index(drop=True),
            df[s == "val"].reset_index(drop=True),
            df[s == "test"].reset_index(drop=True))


def load_geolift() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Meta GeoLift example panels: (pre-test, test). Columns: location, Y, date."""
    download()
    pre = next(iter(pyreadr.read_r(C.DATA_RAW / "GeoLift_PreTest.rda").values()))
    test = next(iter(pyreadr.read_r(C.DATA_RAW / "GeoLift_Test.rda").values()))
    for d in (pre, test):
        d["date"] = pd.to_datetime(d["date"])
    return pre, test
