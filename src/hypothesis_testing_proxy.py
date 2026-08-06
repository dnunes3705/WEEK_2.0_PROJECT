from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def resolve_processed_path(stem: str) -> Path:
    base = Path("data/processed")
    new_path = base / f"{stem}.new.csv"
    old_path = base / f"{stem}.csv"
    return new_path if new_path.exists() else old_path


def fmt(x: float, nd: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "inf" if np.isinf(x) else "NA"
    if isinstance(x, (np.floating,)) and (np.isnan(x) or np.isinf(x)):
        return "inf" if np.isinf(x) else "NA"
    if isinstance(x, float) and np.isnan(x):
        return "NA"
    return f"{x:.{nd}f}"


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int).astype(bool)
    sl = s.astype(str).str.strip().str.lower()
    return sl.isin(["true", "1", "yes", "y", "t"])


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan
    gt = 0
    lt = 0
    for xv in x:
        gt += int((xv > y).sum())
        lt += int((xv < y).sum())
    denom = x.size * y.size
    return float((gt - lt) / denom)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return np.nan
    nx = x.size
    ny = y.size
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


@dataclass(frozen=True)
class TwoGroupSummary:
    metric: str
    n_repeat: int
    n_nonrepeat: int
    mean_repeat: float
    mean_nonrepeat: float
    diff_mean: float
    p_mwu: float
    p_t_welch: float
    cliff_d: float
    cohen_d: float


def two_group_stats(df: pd.DataFrame, metric: str, group_col: str) -> TwoGroupSummary:
    sub = df[[metric, group_col]].dropna().copy()
    g = to_bool_series(sub[group_col])
    x = sub.loc[g == True, metric].to_numpy(dtype=float)
    y = sub.loc[g == False, metric].to_numpy(dtype=float)

    p_mwu = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue) if x.size and y.size else np.nan
    p_t = float(stats.ttest_ind(x, y, equal_var=False).pvalue) if x.size > 1 and y.size > 1 else np.nan

    return TwoGroupSummary(
        metric=metric,
        n_repeat=int(x.size),
        n_nonrepeat=int(y.size),
        mean_repeat=float(np.mean(x)) if x.size else np.nan,
        mean_nonrepeat=float(np.mean(y)) if y.size else np.nan,
        diff_mean=float(np.mean(x) - np.mean(y)) if x.size and y.size else np.nan,
        p_mwu=p_mwu,
        p_t_welch=p_t,
        cliff_d=cliff_delta(x, y),
        cohen_d=cohen_d(x, y),
    )


def logit_table(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    sub = df[[y_col, *x_cols]].dropna().copy()
    y = to_bool_series(sub[y_col]).astype(int)
    X = sub[x_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    params = model.params
    conf = model.conf_int()
    with np.errstate(over="ignore"):
        out = pd.DataFrame(
            {
                "OR": np.exp(params),
                "OR_ci_lo": np.exp(conf[0]),
                "OR_ci_hi": np.exp(conf[1]),
                "p": model.pvalues,
            }
        )
    return out


def poisson_table(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    sub = df[[y_col, *x_cols]].dropna().copy()
    y = sub[y_col].astype(float)
    X = sm.add_constant(sub[x_cols].astype(float), has_constant="add")
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    params = model.params
    conf = model.conf_int()
    with np.errstate(over="ignore"):
        out = pd.DataFrame(
            {
                "RR": np.exp(params),
                "RR_ci_lo": np.exp(conf[0]),
                "RR_ci_hi": np.exp(conf[1]),
                "p": model.pvalues,
            }
        )
    return out


def main() -> int:
    orders = pd.read_csv(resolve_processed_path("orders_features"))
    customers = pd.read_csv(resolve_processed_path("customer_features"))

    customers["repeat_customer_flag"] = to_bool_series(customers["repeat_customer_flag"])
    orders["rated_flag"] = to_bool_series(orders["rated_flag"])

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hypothesis_testing_proxy_scipy.md"

    n_orders = len(orders)
    n_customers = customers["customer_id"].nunique()
    repeat_rate = float(customers["repeat_customer_flag"].mean())

    lines: list[str] = []
    lines.append("# Hypothesis Testing (Proxy Approach - SciPy/Statsmodels)")
    lines.append("")
    lines.append("This report runs proxy hypothesis tests using the available data (no timestamps).")
    lines.append("")
    lines.append("## Data used")
    lines.append(f"- Orders: {n_orders:,}")
    lines.append(f"- Customers: {n_customers:,}")
    lines.append(f"- Repeat customer share (proxy): {repeat_rate:.3f}")
    lines.append("")

    # -----------------------
    # H1
    # -----------------------
    lines.append("## H1 (Proxy): Delivery experience -> repeat customer (proxy)")
    lines.append("Outcome: `repeat_customer_flag` (customer has >=2 orders in dataset).")
    lines.append("")

    h1_metrics = [c for c in ["avg_delivery_time", "avg_total_fulfillment_time", "avg_prep_time"] if c in customers.columns]
    h1 = [two_group_stats(customers, c, "repeat_customer_flag") for c in h1_metrics]

    lines.append("| Metric | n_repeat | n_nonrepeat | mean_repeat | mean_nonrepeat | diff_mean | p_MWU | p_t(Welch) | cliff_d | cohen_d |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in h1:
        lines.append(
            f"| `{r.metric}` | {r.n_repeat:,} | {r.n_nonrepeat:,} | {fmt(r.mean_repeat)} | {fmt(r.mean_nonrepeat)} | {fmt(r.diff_mean)} | {fmt(r.p_mwu)} | {fmt(r.p_t_welch)} | {fmt(r.cliff_d)} | {fmt(r.cohen_d)} |"
        )
    lines.append("")

    # Consistency: n_orders>=2 only, correlate with n_orders
    if "delivery_time_cv" in customers.columns:
        cons = customers.loc[customers["n_orders"] >= 2, ["delivery_time_cv", "n_orders"]].dropna()
        if len(cons) >= 3:
            rho, p = stats.spearmanr(cons["delivery_time_cv"], cons["n_orders"])
            lines.append("Consistency subset: customers with `n_orders >= 2`.")
            lines.append(f"- Spearman(delivery_time_cv, n_orders) = {fmt(float(rho))} (p={fmt(float(p))})")
            lines.append("")

    # Order-level rated only: rating vs time metrics
    rated = orders.loc[orders["rated_flag"] == True].copy()
    if not rated.empty:
        lines.append("Order-level (rated orders): rating vs time metrics (Spearman).")
        for m in ["delivery_time", "total_fulfillment_time_min", "food_preparation_time"]:
            if m not in rated.columns:
                continue
            rho, p = stats.spearmanr(rated["rating_num"], rated[m])
            lines.append(f"- Spearman(rating_num, {m}) = {fmt(float(rho))} (p={fmt(float(p))})")
        lines.append("")

    # H1 Logit
    # Keep controls modest to reduce instability/separation in this snapshot-style proxy setting.
    h1_x = [c for c in ["avg_delivery_time", "avg_prep_time", "avg_cost", "weekend_share"] if c in customers.columns]
    if h1_x:
        lines.append("Logistic regression (proxy): `repeat_customer_flag` ~ delivery + controls (odds ratios).")
        tab = logit_table(customers, "repeat_customer_flag", h1_x).drop(index=["const"], errors="ignore")
        lines.append("| Predictor | OR | OR 95% CI | p |")
        lines.append("|---|---:|---:|---:|")
        for idx, row in tab.iterrows():
            lines.append(f"| `{idx}` | {fmt(float(row['OR']))} | [{fmt(float(row['OR_ci_lo']))}, {fmt(float(row['OR_ci_hi']))}] | {fmt(float(row['p']))} |")
        lines.append("")

    # -----------------------
    # H2
    # -----------------------
    lines.append("## H2 (Proxy): Higher ratings -> higher repeat ordering (proxy)")
    lines.append("Subset: customers with at least one rated order (`has_rated_orders==True`).")
    lines.append("")

    if "has_rated_orders" in customers.columns:
        customers["has_rated_orders"] = to_bool_series(customers["has_rated_orders"])
        cust_rated = customers.loc[customers["has_rated_orders"] == True].copy()
    else:
        cust_rated = customers.loc[customers.get("rated_share", 0) > 0].copy()

    h2_metrics = [c for c in ["avg_rating", "pct_high_rating", "pct_top_rating"] if c in cust_rated.columns]
    h2 = [two_group_stats(cust_rated, c, "repeat_customer_flag") for c in h2_metrics]

    lines.append("| Metric | n_repeat | n_nonrepeat | mean_repeat | mean_nonrepeat | diff_mean | p_MWU | p_t(Welch) | cliff_d | cohen_d |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in h2:
        lines.append(
            f"| `{r.metric}` | {r.n_repeat:,} | {r.n_nonrepeat:,} | {fmt(r.mean_repeat)} | {fmt(r.mean_nonrepeat)} | {fmt(r.diff_mean)} | {fmt(r.p_mwu)} | {fmt(r.p_t_welch)} | {fmt(r.cliff_d)} | {fmt(r.cohen_d)} |"
        )
    lines.append("")

    if "avg_rating" in cust_rated.columns:
        rho, p = stats.spearmanr(cust_rated["avg_rating"], cust_rated["n_orders"], nan_policy="omit")
        lines.append(f"- Spearman(avg_rating, n_orders) among rated customers = {fmt(float(rho))} (p={fmt(float(p))})")
        lines.append("")

    h2_x = [c for c in ["avg_rating", "pct_top_rating", "avg_delivery_time", "avg_cost", "weekend_share"] if c in cust_rated.columns]
    if h2_x:
        lines.append("Logistic regression (proxy, rated customers): `repeat_customer_flag` ~ ratings + controls (odds ratios).")
        tab = logit_table(cust_rated, "repeat_customer_flag", h2_x).drop(index=["const"], errors="ignore")
        lines.append("| Predictor | OR | OR 95% CI | p |")
        lines.append("|---|---:|---:|---:|")
        for idx, row in tab.iterrows():
            lines.append(f"| `{idx}` | {fmt(float(row['OR']))} | [{fmt(float(row['OR_ci_lo']))}, {fmt(float(row['OR_ci_hi']))}] | {fmt(float(row['p']))} |")
        lines.append("")

    # -----------------------
    # H3
    # -----------------------
    lines.append("## H3 (Proxy): Variety -> engagement (proxy)")
    lines.append("Caution: variety grows with `n_orders`, so treat this as descriptive association.")
    lines.append("Proxy approach uses `n_orders` as the engagement outcome.")
    lines.append("")

    for m in ["restaurant_variety", "cuisine_variety"]:
        if m not in customers.columns:
            continue
        rho, p = stats.spearmanr(customers[m], customers["n_orders"], nan_policy="omit")
        lines.append(f"- Spearman({m}, n_orders) = {fmt(float(rho))} (p={fmt(float(p))})")
    lines.append("")

    h3_x = [c for c in ["restaurant_variety", "cuisine_variety", "avg_cost", "avg_delivery_time", "weekend_share"] if c in customers.columns]
    if h3_x:
        lines.append("Poisson regression (proxy): `n_orders` ~ variety + controls (rate ratios).")
        tab = poisson_table(customers, "n_orders", h3_x).drop(index=["const"], errors="ignore")
        lines.append("| Predictor | RR | RR 95% CI | p |")
        lines.append("|---|---:|---:|---:|")
        for idx, row in tab.iterrows():
            lines.append(f"| `{idx}` | {fmt(float(row['RR']))} | [{fmt(float(row['RR_ci_lo']))}, {fmt(float(row['RR_ci_hi']))}] | {fmt(float(row['p']))} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
