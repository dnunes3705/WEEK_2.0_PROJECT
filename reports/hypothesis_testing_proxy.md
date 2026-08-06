# Hypothesis Testing (Proxy Approach)

This report runs **proxy hypothesis tests** using the available data (no timestamps).
All p-values below come from **two-sided permutation tests** (no SciPy/statsmodels required).

## Data used
- Orders: 1,898
- Customers: 1,200
- Repeat customer share (proxy): 0.347

## H1 (Proxy): Delivery experience -> repeat customer (proxy)
Outcome: `repeat_customer_flag` (customer has ≥2 orders in dataset).

| Metric | n_repeat | n_nonrepeat | mean_repeat | mean_nonrepeat | diff (repeat-non) | p (perm) | effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| `avg_delivery_time` | 416 | 784 | 23.929 | 24.356 | -0.427 | 0.118 | -0.089 |
| `avg_total_fulfillment_time` | 416 | 784 | 51.137 | 51.934 | -0.796 | 0.034 | -0.084 |
| `avg_prep_time` | 416 | 784 | 27.208 | 27.578 | -0.369 | 0.145 | -0.051 |

Consistency subset: customers with `n_orders >= 2` (tests `delivery_time_cv` vs `n_orders`).
- Spearman(delivery_time_cv, n_orders) = 0.151 (p_perm=0.002)

Order-level check (rated orders only): `rating_num` vs time metrics (Spearman).
- Spearman(rating_num, delivery_time) = 0.001 (p_perm=0.982)
- Spearman(rating_num, total_fulfillment_time_min) = -0.006 (p_perm=0.851)
- Spearman(rating_num, food_preparation_time) = -0.007 (p_perm=0.822)

## H2 (Proxy): Higher ratings -> higher repeat ordering (proxy)
Subset: customers with at least one rated order (`has_rated_orders==True`).

| Metric | n_repeat | n_nonrepeat | mean_repeat | mean_nonrepeat | diff (repeat-non) | p (perm) | effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| `avg_rating` | 372 | 487 | 4.360 | 4.351 | 0.009 | 0.847 | -0.029 |
| `pct_high_rating` | 372 | 487 | 0.837 | 0.846 | -0.009 | 0.692 | -0.082 |
| `pct_top_rating` | 372 | 487 | 0.523 | 0.505 | 0.018 | 0.573 | 0.013 |

- Spearman(avg_rating, n_orders) among rated customers = -0.035 (p_perm=0.313)

## H3 (Proxy): Variety -> engagement (proxy)
Caution: `repeat_customer_flag` is mechanically linked to `n_orders`, and variety grows with `n_orders`.
Proxy approach here uses **`n_orders`** (count) as the engagement outcome.

- Spearman(restaurant_variety, n_orders) = 0.634 (p_perm=0.000)
- Spearman(cuisine_variety, n_orders) = 0.557 (p_perm=0.000)
