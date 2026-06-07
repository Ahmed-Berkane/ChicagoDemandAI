"""Payroll cycle features derived from the calendar (no external API)."""

from __future__ import annotations

import pandas as pd

BIWEEKLY_PAY_ANCHOR = pd.Timestamp("2025-01-03")


def add_pay_cycle_features(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    d = out[date_col].dt.normalize()

    out["is_semimonthly_payday"] = d.dt.day.isin([1, 15]).astype(int)

    fridays = pd.date_range(
        BIWEEKLY_PAY_ANCHOR,
        d.max() + pd.Timedelta(days=14),
        freq="2W-FRI",
    )
    payday_set = set(fridays.normalize())
    days_since = d.map(
        lambda ts: min(
            ((ts - past).days for past in fridays[fridays <= ts]),
            default=14,
        )
    )
    out["is_payweek"] = (days_since <= 3).astype(int)
    return out


def pay_cycle_flags(target_date: pd.Timestamp) -> dict[str, int]:
    row = add_pay_cycle_features(pd.DataFrame({"date": [target_date.normalize()]}))
    return {
        "is_semimonthly_payday": int(row["is_semimonthly_payday"].iloc[0]),
        "is_payweek": int(row["is_payweek"].iloc[0]),
    }
