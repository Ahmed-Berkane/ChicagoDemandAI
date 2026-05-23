"""Join a real licensed business and its food-inspection history to the daily table."""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_business_licenses, load_food_inspections

LICENSE_COL = "LICENSE NUMBER"
INSPECTION_LICENSE_COL = "License #"
INSPECTION_DATE_COL = "Inspection Date"
RESULTS_COL = "Results"


def select_demo_business(
    licenses: pd.DataFrame | None = None,
    *,
    license_number: int | str | None = None,
    community_area: str = "LOOP",
    coffee_shops_only: bool = True,
) -> pd.Series:
    """
    Pick one active licensed café for the MVP (default: Loop coffee shop with
    the most inspection records in the portal data).
    """
    if licenses is None:
        licenses = load_business_licenses(coffee_shops_only=coffee_shops_only)

    if license_number is not None:
        match = licenses[licenses[LICENSE_COL].astype(str) == str(license_number)]
        if match.empty:
            raise ValueError(f"License {license_number} not found in business licenses.")
        return match.iloc[0]

    pool = licenses[
        licenses["COMMUNITY AREA NAME"].str.upper().str.contains(community_area.upper(), na=False)
    ]
    if pool.empty:
        pool = licenses

    inspections = load_food_inspections()
    inspection_counts = (
        inspections.groupby(INSPECTION_LICENSE_COL)
        .size()
        .rename("n_inspections")
    )
    pool = pool.copy()
    pool["_license_key"] = pd.to_numeric(pool[LICENSE_COL], errors="coerce")
    pool = pool.join(inspection_counts, on="_license_key", how="left")
    pool["n_inspections"] = pool["n_inspections"].fillna(0)

    return pool.sort_values("n_inspections", ascending=False).iloc[0]


def business_static_columns(business: pd.Series) -> dict[str, object]:
    """License fields copied onto every daily row."""
    return {
        "license_number": business[LICENSE_COL],
        "business_name": business.get("DOING BUSINESS AS NAME") or business.get("LEGAL NAME"),
        "business_address": business.get("ADDRESS"),
        "ward": business.get("WARD"),
        "community_area": business.get("COMMUNITY AREA NAME"),
        "neighborhood": business.get("NEIGHBORHOOD"),
        "license_description": business.get("LICENSE DESCRIPTION"),
        "business_latitude": business.get("LATITUDE"),
        "business_longitude": business.get("LONGITUDE"),
    }


def daily_inspection_features(
    start: pd.Timestamp,
    end: pd.Timestamp,
    license_number: int | str,
    inspections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Per-day inspection activity for one license (activity proxy).

    Sparse days are filled with 0 counts.
    """
    if inspections is None:
        inspections = load_food_inspections()

    lic = pd.to_numeric(license_number, errors="coerce")
    biz = inspections[pd.to_numeric(inspections[INSPECTION_LICENSE_COL], errors="coerce") == lic].copy()
    biz["date"] = pd.to_datetime(biz[INSPECTION_DATE_COL], errors="coerce").dt.normalize()

    spine = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    if biz.empty:
        spine["inspections_count"] = 0
        spine["inspections_pass_count"] = 0
        spine["inspections_fail_count"] = 0
        spine["inspections_7d_count"] = 0
        return spine

    biz["is_pass"] = biz[RESULTS_COL].astype(str).str.contains("Pass", case=False, na=False)
    biz["is_fail"] = biz[RESULTS_COL].astype(str).str.contains("Fail", case=False, na=False)

    daily = (
        biz.groupby("date")
        .agg(
            inspections_count=(INSPECTION_DATE_COL, "count"),
            inspections_pass_count=("is_pass", "sum"),
            inspections_fail_count=("is_fail", "sum"),
        )
        .reset_index()
    )
    daily["inspections_pass_count"] = daily["inspections_pass_count"].astype(int)
    daily["inspections_fail_count"] = daily["inspections_fail_count"].astype(int)

    out = spine.merge(daily, on="date", how="left")
    for col in ("inspections_count", "inspections_pass_count", "inspections_fail_count"):
        out[col] = out[col].fillna(0).astype(int)
    out["inspections_7d_count"] = (
        out["inspections_count"].rolling(window=7, min_periods=1).sum().astype(int)
    )
    return out
