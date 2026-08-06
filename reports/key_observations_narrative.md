# Key Observations — Univariate, Bivariate, Multivariate (Narrative)

This narrative summarizes what we observed from the processed feature tables generated from the FoodHub order dataset.

## The dataset in one paragraph

We are working with **1,898 delivered orders** spanning **1,200 customers**, **178 restaurants**, and **14 cuisine types**. Activity is heavily concentrated on weekends: **~71.2% of orders occur on weekends**. Ratings are present for **~61.2% of orders**, with an average of **~4.34** among rated orders; however, **~38.8%** of orders have `rating = "Not given"`, so any rating-based story must account for rating nonresponse.

## What a “typical” order looks like

Orders tend to be modestly priced and fairly consistent in timing:
- **Cost**: mean **$16.50**, median **$14.14** (min **$4.47**, max **$35.41**).
- **Prep time**: mean **27.37 min**, median **27 min** (range **20–35**).
- **Delivery time**: mean **24.16 min**, median **25 min** (range **15–33**).
- **Total fulfillment time** (prep + delivery): mean **51.53 min**, median **52 min** (range **35–68**).

Cuisine demand is concentrated: **American (584 orders)** and **Japanese (470)** lead, followed by **Italian (298)** and **Chinese (215)**.

## Engagement: repeat behavior is meaningful but limited by missing time

Because there are no timestamps, “retention” and “repeat within a time window” cannot be computed as originally defined. Instead, we use within-dataset engagement proxies:
- **~34.7%** of customers are repeat customers (`repeat_customer_flag=True`, i.e., `n_orders >= 2`).
- The average customer places **~1.58 orders** in the observed dataset.

This proxy is still useful for exploratory visuals, but it should be treated as “repeat within the dataset snapshot,” not time-based retention.

## H1 (proxy): delivery experience vs repeat behavior

When we compare customer-level delivery experience between repeat and non-repeat customers, differences exist but are modest:
- Repeat customers have slightly lower typical delivery/fulfillment times (e.g., `avg_delivery_time` median **~24.21** vs **25.00** minutes; `avg_total_fulfillment_time` median **~51.47** vs **52.00** minutes).

Looking at repeat rate by **quartiles of average delivery time** shows a non-monotonic pattern:
- The *slowest* quartile (avg delivery roughly **28–33 min**) has the lowest repeat rate (**~19%**).
- The *fastest* quartile (roughly **15–21 min**) is not the highest repeat rate (**~28%**).
- The highest repeat rate appears in a middle band (roughly **21–24.5 min**) at **~50%**.

Interpretation for later testing: extremely slow delivery looks clearly unfavorable in these proxies, while “fastest possible” is not automatically associated with the highest repeat share—suggesting other factors (restaurant mix, customer segments, order characteristics) may be interacting.

Consistency measures (`delivery_time_cv`, computed only for customers with ≥2 orders) are available for a smaller subset by definition. Among customers with ≥2 orders, the typical variability is moderate (median `delivery_time_cv` **~0.17**), but this should be interpreted cautiously because it excludes one-time customers entirely.

## H2 (proxy): ratings vs repeat behavior (and why this is tricky)

Ratings and repeat are entangled with **how many rated orders** a customer has:
- Customers with only one rated order can have extreme values like `pct_top_rating = 1.0` and `avg_rating = 5.0` while still being non-repeat customers, simply because they only have one observed (rated) order.
- Conversely, customers with multiple rated orders tend to have more “stable” percentages (e.g., `pct_top_rating` values like 0.5, 0.67, etc.).

In visual terms, this means “higher average rating → more repeats” can look inconsistent unless the plots explicitly condition on the number of rated orders (or at least highlight it via point size / facets). For later hypothesis testing, we should either:
- control for `n_orders` / `rated_share` / number of rated orders, or
- compute comparable metrics (e.g., restrict to customers with ≥2 rated orders) to reduce single-observation artifacts.

## H3 (proxy): variety vs repeat behavior (a definitional pitfall)

Variety metrics (`cuisine_variety`, `restaurant_variety`) naturally increase with `n_orders`. This creates a built-in dependency:
- If `cuisine_variety >= 2`, the customer must have placed at least 2 orders, which automatically implies `repeat_customer_flag=True`.

So, the strongest-looking “variety → repeat” patterns are partly definitional rather than behavioral. Variety is still valuable, but for meaningful multivariate visuals (and later modeling), it should be framed against outcomes that aren’t tautological, such as:
- `n_orders` (as a count outcome), or
- variety normalized by order count (e.g., variety per order / entropy-style measures), if we choose to engineer them later.

## Cross-cutting data quality takeaways that shape interpretation

Two structural realities drive many of the “gaps” seen in visuals:
1) **Ratings are missing for ~38.8% of orders** via the sentinel `"Not given"`. This is not random by default, so rating-based conclusions should be treated as conditional on rating being provided.
2) **Variability features** (`std_*`, `*_cv`) are undefined for customers/restaurants with a single order, so missingness there is expected and should be handled via explicit filters (e.g., `n_orders >= 2`).

## What this sets up for next steps

The visual analyses suggest that:
- Very slow deliveries may be associated with lower repeat engagement (proxy), but the relationship is not purely “faster is always better.”
- Rating features need denominator-aware conditioning to avoid single-order artifacts.
- Variety features need careful framing because they are mechanically linked to order count and repeat flags.

When you’re ready, we can translate these observations into a hypothesis-testing-ready dataset slice/definition per hypothesis (still respecting the no-timestamp limitation), without jumping to statistical tests prematurely.

