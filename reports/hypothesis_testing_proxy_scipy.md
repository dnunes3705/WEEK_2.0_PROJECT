# Hypothesis Testing (Proxy Approach - SciPy/Statsmodels)

This report runs proxy hypothesis tests using the available data (no timestamps).

## Data used
- Orders: 1,898
- Customers: 1,200
- Repeat customer share (proxy): 0.347

## H1 (Proxy): Delivery experience -> repeat customer (proxy)
Outcome: `repeat_customer_flag` (customer has >=2 orders in dataset).

| Metric | n_repeat | n_nonrepeat | mean_repeat | mean_nonrepeat | diff_mean | p_MWU | p_t(Welch) | cliff_d | cohen_d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `avg_delivery_time` | 416 | 784 | 23.929 | 24.356 | -0.427 | 0.011 | 0.080 | -0.089 | -0.095 |
| `avg_total_fulfillment_time` | 416 | 784 | 51.137 | 51.934 | -0.796 | 0.016 | 0.017 | -0.084 | -0.129 |
| `avg_prep_time` | 416 | 784 | 27.208 | 27.578 | -0.369 | 0.144 | 0.096 | -0.051 | -0.089 |

Consistency subset: customers with `n_orders >= 2`.
- Spearman(delivery_time_cv, n_orders) = 0.151 (p=0.002)

Order-level (rated orders): rating vs time metrics (Spearman).
- Spearman(rating_num, delivery_time) = 0.001 (p=0.981)
- Spearman(rating_num, total_fulfillment_time_min) = -0.006 (p=0.851)
- Spearman(rating_num, food_preparation_time) = -0.007 (p=0.816)

Logistic regression (proxy): `repeat_customer_flag` ~ delivery + controls (odds ratios).
| Predictor | OR | OR 95% CI | p |
|---|---:|---:|---:|
| `avg_delivery_time` | 0.988 | [0.957, 1.020] | 0.452 |
| `avg_prep_time` | 0.979 | [0.951, 1.008] | 0.149 |
| `avg_cost` | 0.995 | [0.978, 1.014] | 0.626 |
| `weekend_share` | 1.194 | [0.840, 1.698] | 0.322 |

## H2 (Proxy): Higher ratings -> higher repeat ordering (proxy)
Subset: customers with at least one rated order (`has_rated_orders==True`).

| Metric | n_repeat | n_nonrepeat | mean_repeat | mean_nonrepeat | diff_mean | p_MWU | p_t(Welch) | cliff_d | cohen_d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `avg_rating` | 372 | 487 | 4.360 | 4.351 | 0.009 | 0.432 | 0.852 | -0.029 | 0.013 |
| `pct_high_rating` | 372 | 487 | 0.837 | 0.846 | -0.009 | 0.003 | 0.687 | -0.082 | -0.027 |
| `pct_top_rating` | 372 | 487 | 0.523 | 0.505 | 0.018 | 0.713 | 0.568 | 0.013 | 0.038 |

- Spearman(avg_rating, n_orders) among rated customers = -0.035 (p=0.308)

Logistic regression (proxy, rated customers): `repeat_customer_flag` ~ ratings + controls (odds ratios).
| Predictor | OR | OR 95% CI | p |
|---|---:|---:|---:|
| `avg_rating` | 0.832 | [0.533, 1.300] | 0.419 |
| `pct_top_rating` | 1.383 | [0.715, 2.676] | 0.335 |
| `avg_delivery_time` | 0.985 | [0.948, 1.023] | 0.428 |
| `avg_cost` | 0.991 | [0.970, 1.012] | 0.408 |
| `weekend_share` | 1.305 | [0.862, 1.976] | 0.208 |

## H3 (Proxy): Variety -> engagement (proxy)
Caution: variety grows with `n_orders`, so treat this as descriptive association.
Proxy approach uses `n_orders` as the engagement outcome.

- Spearman(restaurant_variety, n_orders) = 0.634 (p=0.000)
- Spearman(cuisine_variety, n_orders) = 0.557 (p=0.000)

Poisson regression (proxy): `n_orders` ~ variety + controls (rate ratios).
| Predictor | RR | RR 95% CI | p |
|---|---:|---:|---:|
| `restaurant_variety` | 1.667 | [1.524, 1.823] | 0.000 |
| `cuisine_variety` | 0.903 | [0.810, 1.006] | 0.065 |
| `avg_cost` | 0.999 | [0.992, 1.006] | 0.786 |
| `avg_delivery_time` | 1.001 | [0.989, 1.014] | 0.817 |
| `weekend_share` | 1.059 | [0.923, 1.215] | 0.417 |
