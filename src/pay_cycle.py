"""Payroll cycle features (synthetic but realistic — no external API needed)."""

from __future__ import annotations

import pandas as pd

# Anchor: a known biweekly Friday payday pattern (adjustable)
BIWEEKLY_PAY_ANCHOR = pd.Timestamp("2025-01-03")


def add_pay_cycle_features(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """
    Add US payroll pattern flags to a daily dataframe.

    - Semimonthly: 1st and 15th (common office payroll)
    - Biweekly: every other Friday from anchor
    - days_since_biweekly_payday: 0 on payday, increases until next
    """
    out = df.copy()
    d = out[date_col].dt.normalize()

    out["is_semimonthly_payday"] = d.dt.day.isin([1, 15])
    out["is_month_end_payday"] = d.dt.is_month_end

    # Biweekly Fridays from anchor
    fridays = pd.date_range(
        BIWEEKLY_PAY_ANCHOR,
        d.max() + pd.Timedelta(days=14),
        freq="2W-FRI",
    )
    payday_set = set(fridays.normalize())
    out["is_biweekly_payday"] = d.isin(payday_set)

    def days_since_payday(ts: pd.Timestamp) -> int:
        past = fridays[fridays <= ts]
        if past.empty:
            return 14
        last = past[-1]
        return (ts - last).days

    out["days_since_biweekly_payday"] = d.map(days_since_payday)
    out["is_payweek"] = out["days_since_biweekly_payday"] <= 3

    return out
