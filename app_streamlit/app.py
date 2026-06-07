"""Chicago Demand Insights — business-facing Streamlit app."""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app_streamlit.business_ui import (
    _rich_text,
    business_insights,
    demand_gauge,
    display_business_category,
    display_region,
    format_forecast_date,
    inject_theme_css,
    render_footer_marker_html,
    render_footer_note_html,
    render_forecast_hero,
    render_header_html,
    render_loading_html,
    render_value_props_html,
    render_welcome_html,
)
from src.inference import (
    EVENT_BOUNDS,
    EVENT_OVERRIDE_COLS,
    SCENARIO_PRESETS,
    WEATHER_BOUNDS,
    WEATHER_OVERRIDE_COLS,
    business_category_options,
    fetch_auto_weather_events,
    predict_demand,
    prediction_date_bounds,
    region_options,
    scenario_presets_for_date,
    season_for_date,
)
from src.modeling.persist import load_artifacts

st.set_page_config(
    page_title="Chicago Demand Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _scenario_cache_key(selected_date: date) -> str:
    return f"scenario_{selected_date.isoformat()}"


def _load_auto_scenario_values(selected_date: date) -> dict[str, float | int]:
    return fetch_auto_weather_events(selected_date)[0]


def _render_business_settings(selected_date: date) -> dict[str, float | int | bool] | None:
    with st.expander(
        "What-if scenarios",
        expanded=st.session_state.get("settings_enabled", False),
        icon=":material/tune:",
    ):
        enabled = st.toggle(
            "Adjust weather & events",
            value=st.session_state.get("settings_enabled", False),
            help="Test how different weather or city events would change your forecast.",
        )
        st.session_state["settings_enabled"] = enabled
        if not enabled:
            return None

        cache_key = _scenario_cache_key(selected_date)
        if cache_key not in st.session_state:
            st.session_state[cache_key] = _load_auto_scenario_values(selected_date)

        season = season_for_date(selected_date)
        seasonal_presets = scenario_presets_for_date(selected_date)
        preset_state_key = f"{cache_key}_preset"
        if st.session_state.get(preset_state_key) not in seasonal_presets:
            st.session_state[preset_state_key] = "Auto (from date)"

        st.caption(f"Season: **{season}** — scenarios below match typical {season.lower()} conditions in Chicago.")
        preset = st.selectbox("Scenario preset", seasonal_presets, key=f"{cache_key}_select")
        if st.button("Reset to auto-detected values", width="stretch"):
            st.session_state[cache_key] = _load_auto_scenario_values(selected_date)
            st.session_state[preset_state_key] = "Auto (from date)"
            st.rerun()

        if st.session_state.get(preset_state_key) != preset:
            if preset == "Auto (from date)":
                st.session_state[cache_key] = _load_auto_scenario_values(selected_date)
            else:
                st.session_state[cache_key] = SCENARIO_PRESETS[preset].copy()
            st.session_state[preset_state_key] = preset

        values = st.session_state[cache_key].copy()
        st.markdown("**Weather**")
        values["temperature_f"] = st.slider(
            "Temperature (F)", *WEATHER_BOUNDS["temperature_f"], float(values.get("temperature_f", 70.0))
        )
        values["precipitation_in"] = st.slider(
            "Rain (in)", *WEATHER_BOUNDS["precipitation_in"], float(values.get("precipitation_in", 0.0)), step=0.05
        )
        values["snowfall_in"] = st.slider(
            "Snow (in)", *WEATHER_BOUNDS["snowfall_in"], float(values.get("snowfall_in", 0.0)), step=0.1
        )
        values["humidity_pct"] = st.slider(
            "Humidity (%)", *WEATHER_BOUNDS["humidity_pct"], float(values.get("humidity_pct", 50.0))
        )
        values["wind_speed_mph"] = st.slider(
            "Wind (mph)", *WEATHER_BOUNDS["wind_speed_mph"], float(values.get("wind_speed_mph", 10.0))
        )

        st.markdown("**Events & sports**")
        city_events = int(round(float(values.get("city_special_events", 0.0))))
        values["city_special_events"] = float(
            st.slider(
                "City events",
                int(EVENT_BOUNDS["city_special_events"][0]),
                int(EVENT_BOUNDS["city_special_events"][1]),
                city_events,
                step=1,
            )
        )
        c1, c2, c3 = st.columns(3)
        values["is_major_festival"] = int(c1.checkbox("Festival", bool(values.get("is_major_festival", 0))))
        values["cubs_home_game"] = int(c2.checkbox("Cubs game", bool(values.get("cubs_home_game", 0))))
        values["bulls_home_game"] = int(c3.checkbox("Bulls game", bool(values.get("bulls_home_game", 0))))

        st.session_state[cache_key] = values
        return {key: values[key] for key in WEATHER_OVERRIDE_COLS + EVENT_OVERRIDE_COLS}
    return None


@st.dialog("Model & Technical Details")
def _technical_details_dialog(metadata: dict, details: dict | None, model_input: pd.DataFrame | None) -> None:
    st.markdown("### Model")
    st.write(
        {
            "model": metadata.get("model_name"),
            "target": metadata.get("target"),
            "encoded_features": len(metadata.get("encoded_features", [])),
            "artifacts": [
                "models/preprocessor.pkl",
                "models/best_model.pkl",
                "models/model_metadata.pkl",
            ],
        }
    )
    st.markdown("### Input features")
    st.code("\n".join(metadata.get("input_features", [])), language="text")
    if details:
        st.markdown("### Last forecast sources")
        st.json(details.get("feature_sources", {}))
        if details.get("overridden_fields"):
            st.write("Overrides:", ", ".join(details["overridden_fields"]))
        st.markdown("### Raw feature row")
        st.json({k: (None if pd.isna(v) else v) for k, v in details.get("raw_row", {}).items()})
    if model_input is not None:
        st.markdown("### Encoded model inputs")
        encoded_inputs = (
            model_input.iloc[0]
            .astype(object)
            .where(model_input.iloc[0].notna(), None)
            .apply(lambda v: str(v) if v is not None else "")
            .reset_index()
        )
        encoded_inputs.columns = ["feature", "value"]
        st.dataframe(encoded_inputs, width="stretch", hide_index=True)


inject_theme_css()

try:
    preprocessor, model, metadata = load_artifacts()
    min_date, max_date = prediction_date_bounds()
except FileNotFoundError:
    st.error("Model files not found. Run `python runner.py` to train and save the model first.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

st.markdown(render_header_html(), unsafe_allow_html=True)

today = date.today()
default_date = min(today + timedelta(days=1), max_date)

with st.container(border=True):
    cfg1, cfg2, cfg3, cfg4 = st.columns([1.1, 1.1, 1.1, 0.9], vertical_alignment="bottom")
    with cfg1:
        selected_date = st.date_input(
            "Forecast date",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
        )
    with cfg2:
        regions = region_options()
        default_region = "DOWNTOWN" if "DOWNTOWN" in regions else regions[0]
        selected_region = st.selectbox(
            "Location",
            regions,
            index=regions.index(default_region),
            format_func=display_region,
        )
    with cfg3:
        categories = business_category_options()
        default_category = "Coffe" if "Coffe" in categories else categories[0]
        selected_category = st.selectbox(
            "Business type",
            categories,
            index=categories.index(default_category),
            format_func=display_business_category,
        )
    with cfg4:
        generate_clicked = st.button("Get Forecast", width="stretch", type="primary")

feature_overrides = _render_business_settings(selected_date)

if generate_clicked:
    st.session_state["forecast_pending"] = {
        "selected_date": selected_date,
        "selected_region": selected_region,
        "selected_category": selected_category,
        "feature_overrides": feature_overrides,
    }
    st.rerun()

forecast_loading = bool(st.session_state.get("forecast_pending"))
if forecast_loading:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(render_loading_html(), unsafe_allow_html=True)
    if not st.session_state.get("forecast_computing"):
        st.session_state["forecast_computing"] = True
        st.rerun()

    pending = st.session_state.pop("forecast_pending")
    st.session_state.pop("forecast_computing", None)
    try:
        prediction, model_input, details = predict_demand(
            pending["selected_date"],
            pending["selected_region"],
            pending["selected_category"],
            feature_overrides=pending["feature_overrides"],
        )
        gauge = demand_gauge(prediction, pending["selected_region"], pending["selected_category"])
        details["reference_median"] = gauge["reference_median"]
        insights = business_insights(
            prediction,
            details,
            gauge,
            business_category=pending["selected_category"],
            region=pending["selected_region"],
            target_date=pending["selected_date"],
        )
        st.session_state["forecast_result"] = {
            "prediction": prediction,
            "gauge": gauge,
            "insights": insights,
            "details": details,
            "model_input": model_input,
            "generated_at": datetime.now(),
            "selected_date": pending["selected_date"],
            "selected_region": pending["selected_region"],
            "selected_category": pending["selected_category"],
        }
    except Exception as exc:
        st.session_state["forecast_error"] = str(exc)
    st.rerun()

if st.session_state.get("forecast_error"):
    st.error(f"Could not generate forecast: {st.session_state.pop('forecast_error')}")

result = st.session_state.get("forecast_result")

if result:
    gauge = result["gauge"]
    insights = result["insights"]
    ref = gauge["reference_median"]
    delta = result["prediction"] - ref
    delta_pct = round((delta / ref) * 100) if ref else 0

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(
        render_forecast_hero(
            result["prediction"],
            gauge,
            region_label=display_region(result["selected_region"]),
            category_label=display_business_category(result["selected_category"]),
            date_label=format_forecast_date(result["selected_date"]),
        ),
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Vs. your typical day", f"{delta_pct:+d}%", delta=f"{delta:+,} customers")
    m2.metric("Demand level", gauge["band"])
    m3.metric("Typical volume", f"{ref:,}", help="Median daily customers for your location and business type")

    st.markdown(f'<div class="summary-callout">{gauge["summary"]}</div>', unsafe_allow_html=True)

    st.markdown("#### What to do")
    insight_cols = st.columns(3)
    cards = [
        ("Staffing", insights["staffing"]),
        ("Inventory & prep", insights["inventory"]),
        ("Day profile", (
            f"**{insights['demand_level']}** demand — {insights['vs_typical']} ({insights['delta_pct']}). "
            f"{insights['day_pattern']}. {insights['profile_note']}"
        )),
    ]
    for col, (title, body) in zip(insight_cols, cards):
        with col:
            st.markdown(
                f'<div class="insight-card"><h4>{title}</h4><p>{_rich_text(body)}</p></div>',
                unsafe_allow_html=True,
            )
elif not forecast_loading:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(render_welcome_html(), unsafe_allow_html=True)
    st.markdown(render_value_props_html(), unsafe_allow_html=True)

generated_at = st.session_state.get("forecast_result", {}).get("generated_at")
refresh_label = (
    f"Updated {generated_at.strftime('%I:%M %p').lstrip('0')}"
    if generated_at
    else "No forecast yet"
)
st.markdown(render_footer_marker_html(), unsafe_allow_html=True)
footer_a, footer_b = st.columns([3, 1])
with footer_a:
    st.markdown(render_footer_note_html(refresh_label), unsafe_allow_html=True)
with footer_b:
    if st.button("Technical details", key="tech_details", type="secondary"):
        last = st.session_state.get("forecast_result")
        _technical_details_dialog(
            metadata,
            last["details"] if last else None,
            last["model_input"] if last else None,
        )
