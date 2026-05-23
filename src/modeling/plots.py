"""Interactive Plotly charts for model evaluation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_actual_vs_predicted_interactive(
    dates: pd.Series,
    y_true,
    y_pred,
    *,
    title: str = "Test set: actual vs predicted demand",
    model_name: str = "",
) -> go.Figure:
    dates = pd.to_datetime(dates)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_true,
            name="Actual traffic",
            mode="lines",
            line=dict(color="#2ecc71", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_pred,
            name="Predicted",
            mode="lines",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Predicted: %{y:.1f}<extra></extra>",
        )
    )
    full_title = f"{title} — {model_name}" if model_name else title
    fig.update_layout(
        title=full_title,
        xaxis_title="Date",
        yaxis_title="customer_traffic",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def plot_feature_importance_interactive(
    importance_df: pd.DataFrame,
    *,
    title: str = "Feature importance (best model)",
    top_n: int = 20,
) -> go.Figure:
    df = importance_df.head(top_n).sort_values("importance", ascending=True)
    source = df["source"].iloc[0] if len(df) else ""

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker_color="#3498db",
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{title}<br><sup>{source}</sup>",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(420, 22 * len(df)),
        template="plotly_white",
    )
    return fig


def plot_importance_comparison(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 15,
) -> go.Figure:
    """Grouped bar when hybrid exposes linear + XGBoost components."""
    df = importance_df.copy()
    top_feats = (
        df.groupby("feature")["importance"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    df = df[df["feature"].isin(top_feats)]
    sources = df["source"].unique()

    fig = make_subplots(rows=1, cols=1)
    colors = {"hybrid (linear |coef|)": "#9b59b6", "hybrid (XGBoost gain)": "#e67e22"}
    for src in sources:
        sub = df[df["source"] == src].sort_values("importance", ascending=True)
        fig.add_trace(
            go.Bar(
                x=sub["importance"],
                y=sub["feature"],
                name=src,
                orientation="h",
                marker_color=colors.get(src, "#3498db"),
            )
        )
    fig.update_layout(
        barmode="group",
        title="Hybrid components — feature importance",
        xaxis_title="Importance",
        height=max(480, 24 * len(top_feats)),
        template="plotly_white",
    )
    return fig
