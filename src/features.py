from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


RATING_NOT_GIVEN = "Not given"


@dataclass(frozen=True)
class FeatureOutputs:
    orders: pd.DataFrame
    customers: pd.DataFrame
    restaurants: pd.DataFrame
    cuisines: pd.DataFrame


def load_orders_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def add_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()

    df["restaurant_name"] = df["restaurant_name"].astype(str).str.strip()
    df["cuisine_type"] = df["cuisine_type"].astype(str).str.strip()
    df["day_of_the_week"] = df["day_of_the_week"].astype(str).str.strip()

    df["is_weekend"] = df["day_of_the_week"].str.lower().eq("weekend")

    rating_raw = df["rating"].astype(str)
    df["rated_flag"] = ~rating_raw.eq(RATING_NOT_GIVEN)
    df["rating_num"] = pd.to_numeric(rating_raw.where(df["rated_flag"]), errors="coerce")

    df["total_fulfillment_time_min"] = df["food_preparation_time"] + df["delivery_time"]
    df["delivery_share_of_total"] = df["delivery_time"] / df["total_fulfillment_time_min"].replace(0, np.nan)

    # Tri-state: unknown when unrated, so group aggregates like mean() reflect "among rated orders".
    df["high_rating_flag"] = df["rating_num"].ge(4).astype("boolean")
    df.loc[~df["rated_flag"], "high_rating_flag"] = pd.NA
    df["top_rating_flag"] = df["rating_num"].eq(5).astype("boolean")
    df.loc[~df["rated_flag"], "top_rating_flag"] = pd.NA

    return df


def _safe_mode(values: pd.Series) -> str | float | int | None:
    if values.empty:
        return None
    modes = values.mode(dropna=True)
    if modes.empty:
        return None
    return modes.iloc[0]


def _share_true(flags: pd.Series) -> float:
    if flags.empty:
        return np.nan
    value = flags.mean(skipna=True)
    if pd.isna(value):
        return np.nan
    return float(value)


def add_customer_features(order_features: pd.DataFrame) -> pd.DataFrame:
    df = order_features.copy()

    grp = df.groupby("customer_id", dropna=False)

    def _std(series: pd.Series) -> float:
        # Sample std (ddof=1); undefined for n<2 -> NaN
        return float(series.std(ddof=1))

    customers = pd.DataFrame(
        {
            "customer_id": grp.size().index,
            "n_orders": grp.size().to_numpy(),
            "repeat_customer_flag": grp.size().ge(2).to_numpy(),
            "has_2plus_orders": grp.size().ge(2).to_numpy(),
            "restaurant_variety": grp["restaurant_name"].nunique().to_numpy(),
            "cuisine_variety": grp["cuisine_type"].nunique().to_numpy(),
            "weekend_share": grp["is_weekend"].mean().to_numpy(),
            "weekday_share": (1.0 - grp["is_weekend"].mean()).to_numpy(),
            "avg_cost": grp["cost_of_the_order"].mean().to_numpy(),
            "median_cost": grp["cost_of_the_order"].median().to_numpy(),
            "std_cost": grp["cost_of_the_order"].apply(_std).to_numpy(),
            "avg_prep_time": grp["food_preparation_time"].mean().to_numpy(),
            "std_prep_time": grp["food_preparation_time"].apply(_std).to_numpy(),
            "avg_delivery_time": grp["delivery_time"].mean().to_numpy(),
            "std_delivery_time": grp["delivery_time"].apply(_std).to_numpy(),
            "avg_total_fulfillment_time": grp["total_fulfillment_time_min"].mean().to_numpy(),
            "std_total_fulfillment_time": grp["total_fulfillment_time_min"].apply(_std).to_numpy(),
            "rated_share": grp["rated_flag"].mean().to_numpy(),
            "has_rated_orders": grp["rated_flag"].any().to_numpy(),
            "avg_rating": grp["rating_num"].mean().to_numpy(),
            "pct_high_rating": grp["high_rating_flag"].apply(_share_true).to_numpy(),
            "pct_top_rating": grp["top_rating_flag"].apply(_share_true).to_numpy(),
            "top_cuisine_type": grp["cuisine_type"].agg(_safe_mode).to_numpy(),
            "top_restaurant_name": grp["restaurant_name"].agg(_safe_mode).to_numpy(),
        }
    )

    customers["delivery_time_cv"] = customers["std_delivery_time"] / customers["avg_delivery_time"].replace(0, np.nan)
    customers["total_time_cv"] = customers["std_total_fulfillment_time"] / customers[
        "avg_total_fulfillment_time"
    ].replace(0, np.nan)

    return customers


def add_restaurant_features(order_features: pd.DataFrame) -> pd.DataFrame:
    df = order_features.copy()
    grp = df.groupby("restaurant_name", dropna=False)

    def _std(series: pd.Series) -> float:
        return float(series.std(ddof=1))

    restaurants = pd.DataFrame(
        {
            "restaurant_name": grp.size().index,
            "cuisine_type": grp["cuisine_type"].agg(_safe_mode).to_numpy(),
            "n_orders": grp.size().to_numpy(),
            "n_customers": grp["customer_id"].nunique().to_numpy(),
            "weekend_share": grp["is_weekend"].mean().to_numpy(),
            "avg_cost": grp["cost_of_the_order"].mean().to_numpy(),
            "avg_prep_time": grp["food_preparation_time"].mean().to_numpy(),
            "avg_delivery_time": grp["delivery_time"].mean().to_numpy(),
            "std_delivery_time": grp["delivery_time"].apply(_std).to_numpy(),
            "avg_total_fulfillment_time": grp["total_fulfillment_time_min"].mean().to_numpy(),
            "rated_share": grp["rated_flag"].mean().to_numpy(),
            "has_rated_orders": grp["rated_flag"].any().to_numpy(),
            "avg_rating": grp["rating_num"].mean().to_numpy(),
            "pct_high_rating": grp["high_rating_flag"].apply(_share_true).to_numpy(),
            "pct_top_rating": grp["top_rating_flag"].apply(_share_true).to_numpy(),
        }
    )

    restaurants["delivery_time_cv"] = restaurants["std_delivery_time"] / restaurants["avg_delivery_time"].replace(
        0, np.nan
    )
    return restaurants


def add_cuisine_features(order_features: pd.DataFrame) -> pd.DataFrame:
    df = order_features.copy()
    grp = df.groupby("cuisine_type", dropna=False)

    cuisines = pd.DataFrame(
        {
            "cuisine_type": grp.size().index,
            "n_orders": grp.size().to_numpy(),
            "n_restaurants": grp["restaurant_name"].nunique().to_numpy(),
            "n_customers": grp["customer_id"].nunique().to_numpy(),
            "weekend_share": grp["is_weekend"].mean().to_numpy(),
            "weekday_share": (1.0 - grp["is_weekend"].mean()).to_numpy(),
            "avg_cost": grp["cost_of_the_order"].mean().to_numpy(),
            "avg_total_fulfillment_time": grp["total_fulfillment_time_min"].mean().to_numpy(),
            "rated_share": grp["rated_flag"].mean().to_numpy(),
            "avg_rating": grp["rating_num"].mean().to_numpy(),
        }
    )
    return cuisines


def build_feature_outputs(raw_orders: pd.DataFrame) -> FeatureOutputs:
    orders = add_order_features(raw_orders)
    customers = add_customer_features(orders)
    restaurants = add_restaurant_features(orders)
    cuisines = add_cuisine_features(orders)
    return FeatureOutputs(orders=orders, customers=customers, restaurants=restaurants, cuisines=cuisines)


def coerce_bool_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype(bool)
    return out
