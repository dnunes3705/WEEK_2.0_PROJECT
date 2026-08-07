from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_NAME = "FoodHub Analysis Report"


def resolve_processed_path(stem: str) -> Path:
    base = Path("data/processed")
    new_path = base / f"{stem}.new.csv"
    old_path = base / f"{stem}.csv"
    return new_path if new_path.exists() else old_path


def fix_mojibake(text: str) -> str:
    # Handles common Windows-1252/UTF-8 mojibake sequences seen in this repo.
    replacements = {
        "â€“": "–",
        "â€”": "—",
        "â‰¥": "≥",
        "â†’": "→",
        "â€™": "’",
        "â€œ": "“",
        "â€": "”",
    }
    out = text
    for bad, good in replacements.items():
        out = out.replace(bad, good)
    return out


def _esc_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def parse_problem_definition(text: str) -> dict[str, str]:
    """
    Parse problem_definition.txt into named sections.

    Expected format:
      1. Business Context
      2. Project Objective
      3. Key Performance Indicators (KPIs)
      4. Key Hypotheses
    """
    raw = fix_mojibake(text).replace("\r\n", "\n").strip()
    # Drop the title line if present
    lines = raw.splitlines()
    if lines and "problem definition" in lines[0].lower():
        raw = "\n".join(lines[1:]).lstrip()

    matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s*(.+)\s*$", raw))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        sections[title] = body
    return sections


def render_problem_statement_html(problem_text: str) -> str:
    sections = parse_problem_definition(problem_text)

    def render_paragraphs(body: str) -> str:
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        return "\n".join(f"<p class=\"p\">{_esc_html(p).replace('\\n', '<br/>')}</p>" for p in paras)

    # KPI parsing (simple)
    kpi_body = sections.get("Key Performance Indicators (KPIs)", "")
    kpi_items: list[tuple[str, str]] = []
    if kpi_body:
        kpi_matches = list(re.finditer(r"(?m)^\s*KPI\s*\d+\s*:\s*(.+)\s*$", kpi_body))
        for i, m in enumerate(kpi_matches):
            kpi_title = m.group(0).strip()
            start = m.end()
            end = kpi_matches[i + 1].start() if i + 1 < len(kpi_matches) else len(kpi_body)
            kpi_desc = kpi_body[start:end].strip()
            kpi_items.append((kpi_title, kpi_desc))

    # Hypothesis parsing (simple)
    hyp_body = sections.get("Key Hypotheses", "")
    hyp_items: list[tuple[str, str]] = []
    if hyp_body:
        hyp_matches = list(re.finditer(r"(?m)^\s*Hypothesis\s*\d+\s*:\s*(.+)\s*$", hyp_body))
        for i, m in enumerate(hyp_matches):
            hyp_title = m.group(0).strip()
            start = m.end()
            end = hyp_matches[i + 1].start() if i + 1 < len(hyp_matches) else len(hyp_body)
            hyp_desc = hyp_body[start:end].strip()
            hyp_items.append((hyp_title, hyp_desc))

    parts: list[str] = []
    # Business Context
    bc = sections.get("Business Context", "")
    if bc:
        parts.append("<div class=\"subsec\">")
        parts.append("<h3 class=\"subsec-title\">Business Context</h3>")
        parts.append(render_paragraphs(bc))
        parts.append("</div>")

    # Project Objective
    po = sections.get("Project Objective", "")
    if po:
        parts.append("<div class=\"subsec\">")
        parts.append("<h3 class=\"subsec-title\">Project Objective</h3>")
        parts.append(render_paragraphs(po))
        parts.append("</div>")

    # KPIs
    if kpi_items:
        parts.append("<div class=\"subsec\">")
        parts.append("<h3 class=\"subsec-title\">Key Performance Indicators (KPIs)</h3>")
        parts.append("<div class=\"kpi-grid\">")
        for title, desc in kpi_items:
            parts.append("<div class=\"kpi-card\">")
            parts.append(f"<div class=\"kpi-card-title\">{_esc_html(title)}</div>")
            parts.append(f"<div class=\"kpi-card-body\">{_esc_html(desc).replace('\\n', '<br/>')}</div>")
            parts.append("</div>")
        parts.append("</div>")
        parts.append("</div>")

    # Hypotheses
    if hyp_items:
        parts.append("<div class=\"subsec\">")
        parts.append("<h3 class=\"subsec-title\">Key Hypotheses</h3>")
        parts.append("<div class=\"kpi-grid\">")
        for title, desc in hyp_items:
            parts.append("<div class=\"kpi-card\">")
            parts.append(f"<div class=\"kpi-card-title\">{_esc_html(title)}</div>")
            parts.append(f"<div class=\"kpi-card-body\">{_esc_html(desc).replace('\\n', '<br/>')}</div>")
            parts.append("</div>")
        parts.append("</div>")
        parts.append("</div>")

    if not parts:
        return f"<p class=\"p\">{_esc_html(fix_mojibake(problem_text)).replace('\\n', '<br/>')}</p>"
    return "\n".join(parts)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


@dataclass(frozen=True)
class SummaryStats:
    n_orders: int
    n_customers: int
    n_restaurants: int
    n_cuisines: int
    weekend_order_share: float
    rated_order_share: float
    avg_rating_rated_orders: float
    avg_total_fulfillment_time_min: float
    repeat_customer_share: float
    avg_orders_per_customer: float


def compute_summary(orders: pd.DataFrame, customers: pd.DataFrame, restaurants: pd.DataFrame, cuisines: pd.DataFrame) -> SummaryStats:
    rated_orders = orders.loc[orders["rated_flag"] == True]
    return SummaryStats(
        n_orders=int(len(orders)),
        n_customers=int(customers["customer_id"].nunique()),
        n_restaurants=int(restaurants["restaurant_name"].nunique()),
        n_cuisines=int(cuisines["cuisine_type"].nunique()),
        weekend_order_share=float(orders["is_weekend"].mean()),
        rated_order_share=float(orders["rated_flag"].mean()),
        avg_rating_rated_orders=float(rated_orders["rating_num"].mean()),
        avg_total_fulfillment_time_min=float(orders["total_fulfillment_time_min"].mean()),
        repeat_customer_share=float(customers["repeat_customer_flag"].mean()),
        avg_orders_per_customer=float(customers["n_orders"].mean()),
    )


def df_to_html_table(df: pd.DataFrame, class_name: str = "tbl") -> str:
    def esc(s: str) -> str:
        return _esc_html(s)

    cols = list(df.columns)
    rows = df.to_numpy().tolist()
    out: list[str] = [f"<table class=\"{class_name}\">"]
    out.append("<thead><tr>" + "".join(f"<th>{esc(str(c))}</th>" for c in cols) + "</tr></thead>")
    out.append("<tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{esc(str(v))}</td>" for v in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def format_table_numbers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
    return out


def render_table_from_md(md_text: str, header: str) -> str:
    # Extract the first markdown table following a header line. Simple but works for our report file.
    # Returns HTML <table>...</table> or "".
    pattern = re.compile(rf"^{re.escape(header)}\s*$", re.MULTILINE)
    m = pattern.search(md_text)
    if not m:
        return ""
    tail = md_text[m.end() :]
    # Find a table: lines starting with |
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    table_lines: list[str] = []
    started = False
    for ln in lines:
        if ln.lstrip().startswith("|"):
            started = True
            table_lines.append(ln.strip())
        elif started:
            break
    if len(table_lines) < 2:
        return ""
    # Parse pipe table
    rows = []
    for ln in table_lines:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        rows.append(parts)
    # Remove alignment row (---)
    rows = [r for r in rows if not all(set(cell) <= set("-: ") for cell in r)]
    if not rows:
        return ""
    header_row = rows[0]
    body_rows = rows[1:]

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    html = ["<table class=\"tbl\">"]
    html.append("<thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in header_row) + "</tr></thead>")
    html.append("<tbody>")
    for r in body_rows:
        html.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def main() -> int:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11

    # Load data (prefer *.new.csv if present)
    orders = pd.read_csv(resolve_processed_path("orders_features"))
    customers = pd.read_csv(resolve_processed_path("customer_features"))
    restaurants = pd.read_csv(resolve_processed_path("restaurant_features"))
    cuisines = pd.read_csv(resolve_processed_path("cuisine_features"))

    # Ensure booleans are proper
    for col in ["is_weekend", "rated_flag", "repeat_customer_flag"]:
        if col in orders.columns:
            orders[col] = orders[col].astype(bool)
        if col in customers.columns:
            customers[col] = customers[col].astype(bool)

    summary = compute_summary(orders, customers, restaurants, cuisines)

    rated_customers = customers.loc[customers["avg_rating"].notna()].copy()
    avg_rating_repeat = float(rated_customers.loc[rated_customers["repeat_customer_flag"] == True, "avg_rating"].mean())
    avg_rating_nonrepeat = float(rated_customers.loc[rated_customers["repeat_customer_flag"] == False, "avg_rating"].mean())

    # -----------------------------
    # Key stats requested for report
    # -----------------------------
    prep_min = int(orders["food_preparation_time"].min())
    prep_max = int(orders["food_preparation_time"].max())
    prep_mean = float(orders["food_preparation_time"].mean())

    top_restaurants = (
        orders["restaurant_name"]
        .value_counts()
        .head(5)
        .rename_axis("restaurant_name")
        .reset_index(name="n_orders")
    )

    weekend_orders = orders.loc[orders["is_weekend"] == True]
    weekend_cuisines = (
        weekend_orders["cuisine_type"]
        .value_counts()
        .head(5)
        .rename_axis("cuisine_type")
        .reset_index(name="weekend_orders")
    )

    pct_cost_gt_20 = float((orders["cost_of_the_order"] > 20).mean())
    mean_delivery_time = float(orders["delivery_time"].mean())

    deliv_by_day = (
        orders.groupby("day_of_the_week", dropna=False)["delivery_time"]
        .agg(n_orders="count", mean="mean", median="median")
        .reset_index()
    )
    # Ensure consistent order if present
    day_order = ["Weekday", "Weekend"]
    if set(day_order).issubset(set(deliv_by_day["day_of_the_week"].astype(str))):
        deliv_by_day["day_of_the_week"] = pd.Categorical(deliv_by_day["day_of_the_week"], categories=day_order, ordered=True)
        deliv_by_day = deliv_by_day.sort_values("day_of_the_week")
    deliv_by_day["mean"] = deliv_by_day["mean"].round(2)
    deliv_by_day["median"] = deliv_by_day["median"].round(2)

    # Promotions (Q13) and revenue (Q14) from learner notebook
    rated_orders = orders.loc[orders["rating"].astype(str) != "Not given"].copy()
    rated_orders["rating_num_tmp"] = pd.to_numeric(rated_orders["rating"], errors="coerce")
    promo = (
        rated_orders.groupby("restaurant_name", dropna=False)["rating_num_tmp"]
        .agg(rating_count="count", avg_rating="mean")
        .reset_index()
    )
    promo = promo[(promo["rating_count"] > 50) & (promo["avg_rating"] > 4)].sort_values(
        ["avg_rating", "rating_count"], ascending=[False, False]
    )
    promo_display = promo.rename(columns={"restaurant_name": "restaurant_name"}).copy()
    promo_display["avg_rating"] = promo_display["avg_rating"].round(3)

    total_revenue = float(
        orders["cost_of_the_order"].apply(lambda x: x * 0.25 if x > 20 else (x * 0.15 if x > 5 else 0.0)).sum()
    )

    top_restaurants_table = df_to_html_table(top_restaurants)
    weekend_cuisines_table = df_to_html_table(weekend_cuisines)
    deliv_by_day_table = df_to_html_table(
        deliv_by_day.rename(
            columns={
                "day_of_the_week": "day_of_the_week",
                "n_orders": "n_orders",
                "mean": "mean_delivery_time",
                "median": "median_delivery_time",
            }
        )
    )
    promo_table = df_to_html_table(format_table_numbers(promo_display))

    reports_dir = Path("reports")
    assets_dir = reports_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Charts
    # -----------------------------
    # 1) Orders by weekday/weekend
    sns.countplot(data=orders, x="day_of_the_week", order=["Weekday", "Weekend"], color="#3b82f6")
    plt.title("Orders by day_of_the_week")
    plt.xlabel("")
    plt.ylabel("# orders")
    save_fig(assets_dir / "orders_by_day.png")

    # 2) Ratings distribution (incl Not given)
    sns.countplot(
        data=orders,
        x="rating",
        hue="rating",
        order=["Not given", "3", "4", "5"],
        palette={"Not given": "#94a3b8", "3": "#f59e0b", "4": "#22c55e", "5": "#16a34a"},
        legend=False,
    )
    plt.title("Orders by rating (including 'Not given')")
    plt.xlabel("")
    plt.ylabel("# orders")
    save_fig(assets_dir / "orders_by_rating.png")

    # 3) Cost distribution
    sns.histplot(data=orders, x="cost_of_the_order", kde=True, bins=30, color="#0ea5e9")
    plt.title("Distribution of cost_of_the_order")
    plt.xlabel("Cost (USD)")
    plt.ylabel("# orders")
    save_fig(assets_dir / "cost_distribution.png")

    # 3b) Cost boxplot
    sns.boxplot(data=orders, x="cost_of_the_order", color="#0ea5e9")
    plt.title("Cost of the order (boxplot)")
    plt.xlabel("Cost (USD)")
    plt.ylabel("")
    save_fig(assets_dir / "cost_boxplot.png")

    # 3c) Cost by cuisine (boxplot)
    cuisine_order_cost = (
        orders.groupby("cuisine_type", dropna=False)["cost_of_the_order"]
        .median()
        .sort_values()
        .index.tolist()
    )
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=orders,
        y="cuisine_type",
        x="cost_of_the_order",
        order=cuisine_order_cost,
        color="#0ea5e9",
    )
    plt.title("Cost of the order by cuisine type")
    plt.xlabel("Cost (USD)")
    plt.ylabel("")
    save_fig(assets_dir / "cost_by_cuisine.png")

    # 4) Fulfillment time distribution
    sns.histplot(data=orders, x="total_fulfillment_time_min", kde=True, bins=30, color="#10b981")
    plt.title("Distribution of total_fulfillment_time_min")
    plt.xlabel("Minutes")
    plt.ylabel("# orders")
    save_fig(assets_dir / "total_time_distribution.png")

    # 4b) Food preparation time histogram
    sns.histplot(data=orders, x="food_preparation_time", kde=True, bins=16, color="#34d399")
    plt.title("Distribution of food_preparation_time")
    plt.xlabel("Minutes")
    plt.ylabel("# orders")
    save_fig(assets_dir / "prep_time_distribution.png")

    # 4c) Food preparation time boxplot
    sns.boxplot(data=orders, x="food_preparation_time", color="#34d399")
    plt.title("food_preparation_time (boxplot)")
    plt.xlabel("Minutes")
    plt.ylabel("")
    save_fig(assets_dir / "prep_time_boxplot.png")

    # 4d) Delivery time histogram (learner notebook style)
    sns.histplot(data=orders, x="delivery_time", kde=True, bins=20, color="#38bdf8")
    plt.title("Distribution of delivery_time")
    plt.xlabel("Minutes")
    plt.ylabel("# orders")
    save_fig(assets_dir / "delivery_time_distribution.png")

    # 4d) Prep time by cuisine (boxplot)
    cuisine_order = (
        orders.groupby("cuisine_type", dropna=False)["food_preparation_time"]
        .median()
        .sort_values()
        .index.tolist()
    )
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=orders,
        y="cuisine_type",
        x="food_preparation_time",
        order=cuisine_order,
        color="#34d399",
    )
    plt.title("Food preparation time by cuisine type")
    plt.xlabel("Minutes")
    plt.ylabel("")
    save_fig(assets_dir / "prep_time_by_cuisine.png")

    # 4e) Delivery time by day_of_the_week (boxplot)
    plt.figure(figsize=(8, 4))
    sns.boxplot(
        data=orders,
        x="day_of_the_week",
        y="delivery_time",
        order=["Weekday", "Weekend"],
        color="#38bdf8",
    )
    plt.title("Delivery time by day_of_the_week")
    plt.xlabel("")
    plt.ylabel("delivery_time (min)")
    save_fig(assets_dir / "delivery_time_by_day_boxplot.png")

    # 4f) Correlation heatmap (learner notebook)
    corr_cols = ["cost_of_the_order", "food_preparation_time", "delivery_time"]
    corr = orders[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
    plt.title("Correlation heatmap (cost, prep, delivery)")
    save_fig(assets_dir / "corr_heatmap_cost_prep_delivery.png")

    # 5b) All cuisine counts (orders)
    cuisine_counts = (
        orders["cuisine_type"]
        .value_counts()
        .rename_axis("cuisine_type")
        .reset_index(name="n_orders")
        .sort_values("n_orders", ascending=False)
    )
    sns.barplot(data=cuisine_counts, y="cuisine_type", x="n_orders", color="#a78bfa")
    plt.title("Cuisine types by order count")
    plt.xlabel("# orders")
    plt.ylabel("")
    save_fig(assets_dir / "cuisine_counts.png")

    # 6) Repeat customer share
    repeat_counts = customers["repeat_customer_flag"].value_counts().rename(index={False: "Non-repeat", True: "Repeat"}).reset_index()
    repeat_counts.columns = ["group", "count"]
    sns.barplot(
        data=repeat_counts,
        x="group",
        y="count",
        hue="group",
        palette={"Non-repeat": "#94a3b8", "Repeat": "#22c55e"},
        legend=False,
    )
    plt.title("Customers: repeat vs non-repeat (proxy)")
    plt.xlabel("")
    plt.ylabel("# customers")
    save_fig(assets_dir / "repeat_share.png")

    # 7) Delivery time by repeat flag
    sns.boxplot(
        data=customers,
        x="repeat_customer_flag",
        y="avg_delivery_time",
        hue="repeat_customer_flag",
        palette={False: "#94a3b8", True: "#22c55e"},
        legend=False,
    )
    plt.title("avg_delivery_time by repeat_customer_flag")
    plt.xlabel("repeat_customer_flag")
    plt.ylabel("Minutes")
    save_fig(assets_dir / "avg_delivery_by_repeat.png")

    # 8) Total time by repeat flag
    sns.boxplot(
        data=customers,
        x="repeat_customer_flag",
        y="avg_total_fulfillment_time",
        hue="repeat_customer_flag",
        palette={False: "#94a3b8", True: "#22c55e"},
        legend=False,
    )
    plt.title("avg_total_fulfillment_time by repeat_customer_flag")
    plt.xlabel("repeat_customer_flag")
    plt.ylabel("Minutes")
    save_fig(assets_dir / "avg_total_by_repeat.png")

    # 9) Scatter: n_orders vs avg_delivery_time
    plot_df = customers.copy()
    plot_df["n_orders_jitter"] = plot_df["n_orders"] + np.random.default_rng(7).uniform(-0.08, 0.08, size=len(plot_df))
    sns.scatterplot(data=plot_df, x="avg_delivery_time", y="n_orders_jitter", alpha=0.35, color="#0ea5e9")
    plt.title("avg_delivery_time vs n_orders (jittered)")
    plt.xlabel("avg_delivery_time (min)")
    plt.ylabel("n_orders")
    save_fig(assets_dir / "delivery_vs_orders.png")

    # 9b) Point plots from learner notebook (order-level)
    rating_order = ["Not given", "3", "4", "5"]

    plt.figure(figsize=(10, 4))
    sns.pointplot(x="rating", y="delivery_time", data=orders, order=rating_order, color="#38bdf8")
    plt.title("Relationship between rating and delivery time")
    plt.xlabel("rating")
    plt.ylabel("delivery_time (min)")
    save_fig(assets_dir / "point_rating_delivery_time.png")

    plt.figure(figsize=(10, 4))
    sns.pointplot(x="rating", y="food_preparation_time", data=orders, order=rating_order, color="#34d399")
    plt.title("Relationship between rating and food preparation time")
    plt.xlabel("rating")
    plt.ylabel("food_preparation_time (min)")
    save_fig(assets_dir / "point_rating_prep_time.png")

    plt.figure(figsize=(10, 4))
    sns.pointplot(x="rating", y="cost_of_the_order", data=orders, order=rating_order, color="#f59e0b")
    plt.title("Relationship between rating and cost of the order")
    plt.xlabel("rating")
    plt.ylabel("cost_of_the_order (USD)")
    save_fig(assets_dir / "point_rating_cost.png")

    # 10) Avg rating by repeat flag (rated customers only)
    rated_customers = customers.loc[customers.get("has_rated_orders", customers.get("rated_share", 0) > 0) == True].copy()
    if "avg_rating" in rated_customers.columns and not rated_customers.empty:
        sns.boxplot(
            data=rated_customers,
            x="repeat_customer_flag",
            y="avg_rating",
            hue="repeat_customer_flag",
            palette={False: "#94a3b8", True: "#22c55e"},
            legend=False,
        )
        plt.title("avg_rating by repeat_customer_flag (customers with ratings)")
        plt.xlabel("repeat_customer_flag")
        plt.ylabel("avg_rating")
        save_fig(assets_dir / "avg_rating_by_repeat.png")

    # 11) Heatmap: repeat rate by delivery quartile x cost quartile
    heat = customers.copy()
    heat["avg_delivery_time_q"] = pd.qcut(heat["avg_delivery_time"], q=4, duplicates="drop")
    heat["avg_cost_q"] = pd.qcut(heat["avg_cost"], q=4, duplicates="drop")
    pivot = (
        heat.groupby(["avg_delivery_time_q", "avg_cost_q"], dropna=False)["repeat_customer_flag"]
        .mean()
        .reset_index(name="repeat_rate")
        .pivot(index="avg_delivery_time_q", columns="avg_cost_q", values="repeat_rate")
    )
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Repeat share by avg_delivery_time quartile × avg_cost quartile")
    plt.xlabel("avg_cost quartile")
    plt.ylabel("avg_delivery_time quartile")
    save_fig(assets_dir / "repeat_heatmap_delivery_cost.png")

    # -----------------------------
    # Hypothesis testing tables (already generated)
    # -----------------------------
    ht_path = reports_dir / "hypothesis_testing_proxy_scipy.md"
    ht_text = ht_path.read_text(encoding="utf-8") if ht_path.exists() else ""
    ht_text = fix_mojibake(ht_text)

    h1_tbl = render_table_from_md(ht_text, "## H1 (Proxy): Delivery experience -> repeat customer (proxy)")
    h1_logit = render_table_from_md(ht_text, "Logistic regression (proxy): `repeat_customer_flag` ~ delivery + controls (odds ratios).")
    h2_tbl = render_table_from_md(ht_text, "## H2 (Proxy): Higher ratings -> higher repeat ordering (proxy)")
    h2_logit = render_table_from_md(ht_text, "Logistic regression (proxy, rated customers): `repeat_customer_flag` ~ ratings + controls (odds ratios).")
    h3_pois = render_table_from_md(ht_text, "Poisson regression (proxy): `n_orders` ~ variety + controls (rate ratios).")

    # -----------------------------
    # Narrative inputs
    # -----------------------------
    problem_text = fix_mojibake(Path("problem_definition.txt").read_text(encoding="utf-8"))
    problem_html = render_problem_statement_html(problem_text)

    # -----------------------------
    # HTML
    # -----------------------------
    def esc(s: str) -> str:
        return _esc_html(s)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{PROJECT_NAME}</title>
    <style>
    :root {{
      --bg: #0b1220;
      --panel: #0f172a;
      --card: #0b1b33;
      --border: rgba(255,255,255,.10);
      --text: rgba(255,255,255,.92);
      --muted: rgba(255,255,255,.72);
      --muted2: rgba(255,255,255,.55);
      --accent: #38bdf8;
      --accent2: #22c55e;
      --warn: #f59e0b;
      --shadow: 0 10px 24px rgba(0,0,0,.35);
      --radius: 14px;
      --maxw: 1120px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(1200px 700px at 15% 10%, rgba(56,189,248,.18), transparent 60%),
        radial-gradient(900px 600px at 90% 0%, rgba(34,197,94,.14), transparent 55%),
        linear-gradient(180deg, var(--bg), #070b14);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    .wrap {{ max-width: var(--maxw); margin: 0 auto; padding: 28px 18px 72px; }}
    .hero {{
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(15,23,42,.88), rgba(11,27,51,.55));
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px 22px 18px;
    }}
    .hero h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: .2px; }}
    .hero p {{ margin: 0; color: var(--muted); line-height: 1.5; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .chip {{
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
      border: 1px solid var(--border);
      background: rgba(255,255,255,.04);
      padding: 6px 10px;
      border-radius: 999px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
      margin-top: 16px;
    }}
    .card {{
      grid-column: span 3;
      border: 1px solid var(--border);
      background: rgba(15,23,42,.72);
      border-radius: var(--radius);
      padding: 14px 14px 12px;
      box-shadow: 0 8px 18px rgba(0,0,0,.22);
    }}
    .key-stats-grid {{
      grid-template-columns: repeat(5, minmax(0, 1fr));
      margin-top: 0;
    }}
    .key-stats-grid .card {{ grid-column: auto; }}
    .kpi-label {{ color: var(--muted2); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .kpi-value {{ font-size: 22px; margin-top: 6px; }}
    .kpi-sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    section {{ margin-top: 22px; }}
    .section-title {{
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: .2px;
    }}
    .panel {{
      border: 1px solid var(--border);
      background: rgba(15,23,42,.55);
      border-radius: var(--radius);
      padding: 18px;
      box-shadow: 0 8px 18px rgba(0,0,0,.18);
    }}
    .p {{
      margin: 0 0 10px;
      color: var(--muted);
      line-height: 1.65;
    }}
    .subsec {{ margin: 0 0 16px; }}
    .subsec-title {{
      margin: 0 0 8px;
      font-size: 15px;
      letter-spacing: .2px;
      color: rgba(255,255,255,.92);
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-top: 10px;
    }}
    .kpi-card {{
      border: 1px solid var(--border);
      background: rgba(255,255,255,.03);
      border-radius: 12px;
      padding: 12px 12px 10px;
    }}
    .kpi-card-title {{
      font-family: var(--mono);
      font-size: 12px;
      color: rgba(255,255,255,.86);
      margin-bottom: 6px;
    }}
    .kpi-card-body {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .cols {{ display: grid; grid-template-columns: 1.2fr .8fr; gap: 14px; }}
    .note {{
      border-left: 3px solid var(--warn);
      background: rgba(245,158,11,.08);
      padding: 10px 12px;
      border-radius: 10px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .imgcard {{
      border: 1px solid var(--border);
      background: rgba(255,255,255,.03);
      border-radius: var(--radius);
      padding: 10px;
      overflow: hidden;
    }}
    .imgcard img {{ width: 100%; display: block; border-radius: 10px; background: white; }}
    .imgcap {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .two {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .three {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 14px;
    }}
    .spacer {{ height: 14px; }}
    ul {{ margin: 10px 0 0 18px; color: var(--muted); line-height: 1.55; }}
    code {{ font-family: var(--mono); font-size: 0.95em; color: rgba(255,255,255,.86); }}
    pre {{
      background: rgba(0,0,0,.25);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      overflow-x: auto;
      color: rgba(255,255,255,.86);
      line-height: 1.45;
    }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.18);
    }}
    .tbl th, .tbl td {{
      padding: 10px 10px;
      border-bottom: 1px solid rgba(255,255,255,.08);
      font-size: 13px;
      color: rgba(255,255,255,.88);
      vertical-align: top;
    }}
    .tbl th {{
      text-align: left;
      font-size: 12px;
      color: var(--muted2);
      text-transform: uppercase;
      letter-spacing: .08em;
      background: rgba(255,255,255,.03);
    }}
    .tbl tr:last-child td {{ border-bottom: none; }}
    .footer {{
      margin-top: 28px;
      color: var(--muted2);
      font-size: 12px;
      text-align: center;
    }}
    @media (max-width: 980px) {{
      .card {{ grid-column: span 6; }}
      .cols {{ grid-template-columns: 1fr; }}
      .two {{ grid-template-columns: 1fr; }}
      .three {{ grid-template-columns: 1fr; }}
      .kpi-grid {{ grid-template-columns: 1fr; }}
      .key-stats-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 560px) {{
      .key-stats-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>{PROJECT_NAME}</h1>
      <p>A consolidated report covering the problem statement, data overview, exploratory analysis, proxy hypothesis testing, key insights, and recommendations.</p>
      <div class="chips">
        <div class="chip">Source: data/raw/foodhub_order.csv</div>
        <div class="chip">Processed: data/processed/*_features(.new).csv</div>
        <div class="chip">Generated by: <code>python src/build_html_report.py</code></div>
      </div>
      <div class="grid">
        <div class="card">
          <div class="kpi-label">Orders</div>
          <div class="kpi-value">{summary.n_orders:,}</div>
          <div class="kpi-sub">Delivered orders in dataset</div>
        </div>
        <div class="card">
          <div class="kpi-label">Customers</div>
          <div class="kpi-value">{summary.n_customers:,}</div>
          <div class="kpi-sub">Unique customers observed</div>
        </div>
        <div class="card">
          <div class="kpi-label">Weekend Share</div>
          <div class="kpi-value">{summary.weekend_order_share:.1%}</div>
          <div class="kpi-sub">Orders labeled Weekend</div>
        </div>
        <div class="card">
          <div class="kpi-label">Rated Orders</div>
          <div class="kpi-value">{summary.rated_order_share:.1%}</div>
          <div class="kpi-sub">Rating present (not 'Not given')</div>
        </div>
      </div>
    </div>

    <section>
      <h2 class="section-title">Problem Statement</h2>
      <div class="panel">
        {problem_html}
        <div class="note" style="margin-top:12px;">
          <strong>Key limitation</strong><br/>
          This dataset has no <code>order_datetime</code>. Retention and time-window KPIs cannot be computed as defined, so analyses use within-dataset engagement proxies such as <code>repeat_customer_flag</code> and <code>n_orders</code>.
        </div>

        <div style="margin-top:14px;">
          <h3 class="subsec-title" style="margin-bottom:10px;">Key Stats (Quick Facts)</h3>
          <div class="grid key-stats-grid">
            <div class="card">
              <div class="kpi-label">Prep Time (min)</div>
              <div class="kpi-value">{prep_mean:.2f}</div>
              <div class="kpi-sub">Avg (min={prep_min}, max={prep_max})</div>
            </div>
            <div class="card">
              <div class="kpi-label">Mean Delivery Time</div>
              <div class="kpi-value">{mean_delivery_time:.2f}</div>
              <div class="kpi-sub">Minutes (order-level)</div>
            </div>
            <div class="card">
              <div class="kpi-label">Mean Fulfillment Time</div>
              <div class="kpi-value">{summary.avg_total_fulfillment_time_min:.2f}</div>
              <div class="kpi-sub">Minutes (prep + delivery)</div>
            </div>
            <div class="card">
              <div class="kpi-label">Orders &gt; $20</div>
              <div class="kpi-value">{pct_cost_gt_20:.1%}</div>
              <div class="kpi-sub">Share of all orders</div>
            </div>
            <div class="card">
              <div class="kpi-label">Weekend Orders</div>
              <div class="kpi-value">{summary.weekend_order_share:.1%}</div>
              <div class="kpi-sub">Share of all orders</div>
            </div>
          </div>

          <div class="two" style="margin-top:14px;">
            <div class="imgcard">
              <div class="imgcap" style="margin-top:0;margin-bottom:8px;"><strong>Top 5 restaurants</strong> (by # of orders)</div>
              {top_restaurants_table}
            </div>
            <div class="imgcard">
              <div class="imgcap" style="margin-top:0;margin-bottom:8px;"><strong>Most popular cuisines on weekends</strong> (top 5)</div>
              {weekend_cuisines_table}
            </div>
          </div>

          <div class="imgcard" style="margin-top:14px;">
            <div class="imgcap" style="margin-top:0;margin-bottom:8px;"><strong>Delivery time: weekdays vs weekends</strong></div>
            {deliv_by_day_table}
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Data Overview</h2>
      <div class="panel">
        <div class="three">
          <div class="imgcard">
            <img src="assets/orders_by_day.png" alt="Orders by day of the week" />
            <div class="imgcap">Orders are heavily weekend-weighted.</div>
          </div>
          <div class="imgcard">
            <img src="assets/orders_by_rating.png" alt="Orders by rating" />
            <div class="imgcap">A large share of orders have no rating (“Not given”).</div>
          </div>
          <div class="imgcard">
            <img src="assets/cuisine_counts.png" alt="Cuisine counts" />
            <div class="imgcap">Cuisine types ranked by number of orders.</div>
          </div>
        </div>

        <div class="spacer"></div>

        <div class="two">
          <div class="imgcard">
            <img src="assets/cost_distribution.png" alt="Cost distribution" />
            <div class="imgcap">Order cost distribution (USD).</div>
          </div>
          <div class="imgcard">
            <img src="assets/cost_boxplot.png" alt="Cost boxplot" />
            <div class="imgcap">Cost distribution summary (median, IQR, outliers).</div>
          </div>
        </div>

        <div class="spacer"></div>

        <div class="two">
          <div class="imgcard">
            <img src="assets/cost_by_cuisine.png" alt="Cost of the order by cuisine type" />
            <div class="imgcap">Cost of the order by cuisine type (boxplot).</div>
          </div>
          <div class="imgcard">
            <img src="assets/prep_time_distribution.png" alt="Food preparation time distribution" />
            <div class="imgcap">Food preparation time distribution (minutes).</div>
          </div>
        </div>

        <div class="spacer"></div>

        <div class="two">
          <div class="imgcard">
            <img src="assets/prep_time_boxplot.png" alt="Food preparation time boxplot" />
            <div class="imgcap">Food preparation time summary (median, IQR, outliers).</div>
          </div>
          <div class="imgcard">
            <img src="assets/prep_time_by_cuisine.png" alt="Food preparation time by cuisine type" />
            <div class="imgcap">Food preparation time by cuisine type (boxplot).</div>
          </div>
        </div>

        <div class="spacer"></div>

        <div class="two">
          <div class="imgcard">
            <img src="assets/total_time_distribution.png" alt="Total fulfillment time distribution" />
            <div class="imgcap">Prep+delivery time distribution (minutes).</div>
          </div>
          <div class="imgcard">
            <img src="assets/delivery_time_by_day_boxplot.png" alt="Delivery time by day of the week (boxplot)" />
            <div class="imgcap">Delivery time by day_of_the_week (boxplot).</div>
          </div>
        </div>

        <div class="spacer"></div>

        <div class="two">
          <div class="imgcard">
            <img src="assets/delivery_time_distribution.png" alt="Delivery time distribution" />
            <div class="imgcap">Delivery time histogram (minutes).</div>
          </div>
          <div class="imgcard">
            <img src="assets/corr_heatmap_cost_prep_delivery.png" alt="Correlation heatmap" />
            <div class="imgcap">Heatmap of correlations among cost, prep time, and delivery time.</div>
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Hypothetical Promotions & Revenue</h2>
      <div class="panel cols">
        <div>
          <h3 class="subsec-title">Promotion-eligible restaurants (Q13)</h3>
          <p class="p">Eligibility rule: <code>rating_count &gt; 50</code> and <code>avg_rating &gt; 4</code> (ratings exclude <code>Not given</code>).</p>
          {promo_table if len(promo_display) else '<div class="note">No restaurants meet the criteria in this dataset snapshot.</div>'}
        </div>
        <div class="note">
          <strong>Net revenue estimate (Q14)</strong><br/>
          <div style="margin-top:8px; color: var(--muted); line-height:1.55;">
            Fee rule applied per order:
            <ul style="margin-top:6px;">
              <li>25% of <code>cost_of_the_order</code> if cost &gt; $20</li>
              <li>15% of <code>cost_of_the_order</code> if cost &gt; $5 (and ≤ $20)</li>
              <li>0% otherwise</li>
            </ul>
            <div style="margin-top:10px;">
              <span class="kpi-label">Net revenue (all orders)</span><br/>
              <span style="font-size:22px; font-weight:600; color: rgba(255,255,255,.92);">${total_revenue:,.2f}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Engagement & Customer Segments (Proxy)</h2>
      <div class="panel two">
        <div class="imgcard">
          <img src="assets/repeat_share.png" alt="Repeat share" />
          <div class="imgcap">Repeat customers are those with <code>n_orders >= 2</code> in the dataset snapshot.</div>
        </div>
        <div class="imgcard">
          <img src="assets/delivery_vs_orders.png" alt="Delivery time vs orders" />
          <div class="imgcap">Relationship between delivery speed and observed order count (jittered).</div>
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Hypothesis-Driven Visual Analysis</h2>
      <div class="panel">
        <h3 style="margin:0 0 10px;">H1: Delivery experience → repeat behavior (proxy)</h3>
        <div class="two">
          <div class="imgcard">
            <img src="assets/avg_delivery_by_repeat.png" alt="Avg delivery time by repeat flag" />
            <div class="imgcap">Repeat vs non-repeat customers by average delivery time.</div>
          </div>
          <div class="imgcard">
            <img src="assets/avg_total_by_repeat.png" alt="Avg total fulfillment time by repeat flag" />
            <div class="imgcap">Repeat vs non-repeat customers by total fulfillment time.</div>
          </div>
        </div>
        <div class="imgcard" style="margin-top:14px;">
          <img src="assets/repeat_heatmap_delivery_cost.png" alt="Heatmap repeat rate by delivery and cost quartiles" />
          <div class="imgcap">Repeat share across delivery-time quartiles and customer spend quartiles (descriptive).</div>
        </div>

        <h3 style="margin:18px 0 10px;">H2: Ratings → repeat behavior (proxy)</h3>
        <div class="two">
          <div class="imgcard">
            <img src="assets/avg_rating_by_repeat.png" alt="Avg rating by repeat flag" />
            <div class="imgcap">Ratings are shown only for customers with at least one rated order.</div>
          </div>
          <div class="note">
            <strong>Interpretation caveat</strong><br/>
            Many customers have only one rated order, which can create extreme values (e.g., <code>pct_top_rating=1.0</code>) without indicating a stable long-run tendency.
          </div>
        </div>
        <div class="two" style="margin-top:14px;">
          <div class="imgcard">
            <img src="assets/point_rating_delivery_time.png" alt="Point plot rating vs delivery time" />
            <div class="imgcap">Point plot: rating vs delivery time (order-level).</div>
          </div>
          <div class="imgcard">
            <img src="assets/point_rating_prep_time.png" alt="Point plot rating vs food preparation time" />
            <div class="imgcap">Point plot: rating vs food preparation time (order-level).</div>
          </div>
        </div>
        <div class="imgcard" style="margin-top:14px;">
          <img src="assets/point_rating_cost.png" alt="Point plot rating vs cost of the order" />
          <div class="imgcap">Point plot: rating vs cost of the order (order-level).</div>
        </div>

        <h3 style="margin:18px 0 10px;">H3: Variety → engagement (proxy)</h3>
        <div class="note">
          Variety measures (e.g., <code>cuisine_variety</code>) naturally grow with <code>n_orders</code>. With no timestamps, “variety → retention” is best treated as descriptive and should avoid leakage in future time-based analyses.
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Proxy Hypothesis Testing Results (SciPy/Statsmodels)</h2>
      <div class="panel">
        <p style="margin:0 0 12px;color:var(--muted);line-height:1.6;">
          These results use proxy outcomes because the dataset lacks timestamps. Use them as directional evidence, not definitive retention measurement.
        </p>
        <h3 style="margin:0 0 8px;">H1 summary</h3>
        {h1_tbl or '<p class="note">Could not parse H1 table from the markdown report.</p>'}
        <div style="margin-top:12px;"></div>
        {h1_logit or ''}

        <h3 style="margin:18px 0 8px;">H2 summary</h3>
        {h2_tbl or '<p class="note">Could not parse H2 table from the markdown report.</p>'}
        <div style="margin-top:12px;"></div>
        {h2_logit or ''}

        <h3 style="margin:18px 0 8px;">H3 summary</h3>
        {h3_pois or '<p class="note">Could not parse H3 Poisson table from the markdown report.</p>'}
      </div>
    </section>

    <section>
      <h2 class="section-title">Key Insights</h2>
      <div class="panel">
        <ul>
          <li><strong>Weekend-heavy demand:</strong> ~{summary.weekend_order_share:.0%} of orders occur on weekends, suggesting staffing and partner capacity decisions should prioritize weekend performance.</li>
          <li><strong>Ratings are incomplete:</strong> ~{(1-summary.rated_order_share):.0%} of orders are unrated (“Not given”), so rating-based insights are conditional on customers choosing to rate.</li>
          <li><strong>Ratings vs repeat (proxy):</strong> among customers who rated, average customer rating is ~{avg_rating_repeat:.2f} (repeat) vs ~{avg_rating_nonrepeat:.2f} (non-repeat), suggesting little separation in this dataset.</li>
          <li><strong>Typical end-to-end fulfillment:</strong> ~{summary.avg_total_fulfillment_time_min:.0f} minutes on average from prep start to delivery (prep + delivery).</li>
          <li><strong>Delivery time signal is subtle:</strong> repeat customers show slightly lower average fulfillment times, but simple models show limited explanatory power without richer context (distance, location, timestamps).</li>
          <li><strong>Variety relates strongly to order count:</strong> variety measures are mechanically linked to order volume; time-based windows are needed to test causality cleanly.</li>
        </ul>
      </div>
    </section>

    <section>
      <h2 class="section-title">Recommendations</h2>
      <div class="panel cols">
        <div>
          <h3 style="margin:0 0 8px;">Operational</h3>
          <ul>
            <li>Target the slowest deliveries first: identify restaurants/cuisines with long delivery tails and reduce variance (dispatch, batching rules, partner SLAs).</li>
            <li>Plan for weekend load: staffing and courier supply should reflect the strong weekend skew.</li>
          </ul>
          <h3 style="margin:14px 0 8px;">Product / Measurement</h3>
          <ul>
            <li>Increase rating capture (without bias): prompt at consistent points post-delivery and monitor rating-response rates by segment.</li>
            <li>Add timestamps (<code>order_datetime</code>) and a clear retention window definition so KPI1–KPI3 can be measured as stated.</li>
            <li>Request additional operational data (pickup/dropoff timestamps, delivery distance/zone, courier &amp; batching indicators) to better diagnose delays and drivers of repeat usage.</li>
          </ul>
        </div>
        <div class="note">
          <strong>Data to request next</strong><br/>
          <ul style="margin-top:8px;">
            <li><code>order_datetime</code> (or at least <code>order_date</code>)</li>
            <li>Pickup/dropoff timestamps (or delivery pipeline events)</li>
            <li>Delivery distance / zone (borough)</li>
            <li>Courier identifier and batch indicators</li>
          </ul>
        </div>
      </div>
    </section>

    <div class="footer">
      Generated locally. Output: <code>reports/foodhub_report.html</code>
    </div>
  </div>
</body>
</html>
"""

    out_html = reports_dir / "foodhub_report.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
