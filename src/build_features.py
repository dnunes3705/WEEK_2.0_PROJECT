from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.features import build_feature_outputs, load_orders_csv


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        # Common on Windows when the CSV is open in Excel/preview.
        fallback = path.with_name(f"{path.stem}.new{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"Warning: could not overwrite `{path}` (PermissionError). Wrote `{fallback}` instead.")


def _render_descriptive_report(outputs, report_path: Path, raw_path: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    orders = outputs.orders
    customers = outputs.customers
    restaurants = outputs.restaurants
    cuisines = outputs.cuisines

    lines: list[str] = []
    lines.append("# Descriptive Summary (Engineered Features)")
    lines.append("")
    lines.append(f"Source: `{raw_path}`")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Orders: {len(orders):,}")
    lines.append(f"- Customers: {customers['customer_id'].nunique():,}")
    lines.append(f"- Restaurants: {restaurants['restaurant_name'].nunique():,}")
    lines.append(f"- Cuisines: {cuisines['cuisine_type'].nunique():,}")
    lines.append("")
    lines.append("## Rating Coverage")
    lines.append(f"- Rated share (orders): {orders['rated_flag'].mean():.3f}")
    lines.append(f"- Avg rating (rated orders): {orders.loc[orders['rated_flag'], 'rating_num'].mean():.3f}")
    lines.append("")
    lines.append("## Time (minutes)")
    lines.append(f"- Avg prep time: {orders['food_preparation_time'].mean():.2f}")
    lines.append(f"- Avg delivery time: {orders['delivery_time'].mean():.2f}")
    lines.append(f"- Avg total fulfillment time: {orders['total_fulfillment_time_min'].mean():.2f}")
    lines.append("")
    lines.append("## Engagement Proxies (no timestamps)")
    lines.append(f"- Repeat customer share: {customers['repeat_customer_flag'].mean():.3f}")
    lines.append(f"- Avg orders per customer: {customers['n_orders'].mean():.3f}")
    lines.append("")
    lines.append("## Top Cuisines (by orders)")
    top_cuisines = cuisines.sort_values("n_orders", ascending=False).head(10)[["cuisine_type", "n_orders"]]
    for _, row in top_cuisines.iterrows():
        lines.append(f"- {row['cuisine_type']}: {int(row['n_orders']):,}")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build engineered features for FoodHub EDA.")
    parser.add_argument("--raw", default="data/raw/foodhub_order.csv", help="Path to raw orders CSV")
    parser.add_argument("--outdir", default="data/processed", help="Output directory for feature CSVs")
    parser.add_argument("--report", default="reports/descriptive_summary.md", help="Path for markdown summary report")
    args = parser.parse_args()

    raw_path = args.raw
    outdir = Path(args.outdir)
    report_path = Path(args.report)

    raw_orders = load_orders_csv(raw_path)
    outputs = build_feature_outputs(raw_orders)

    _write_csv(outputs.orders, outdir / "orders_features.csv")
    _write_csv(outputs.customers, outdir / "customer_features.csv")
    _write_csv(outputs.restaurants, outdir / "restaurant_features.csv")
    _write_csv(outputs.cuisines, outdir / "cuisine_features.csv")

    _render_descriptive_report(outputs, report_path, raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
