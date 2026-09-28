-- Arm-level readout, plus the pooled "Any E-Mail" group. Counts and moments only:
-- every interval/test is computed in Python from these sufficient statistics.
WITH grouped AS (
    SELECT arm, segment AS grp, visit, conversion, spend FROM customers
    UNION ALL
    SELECT -1 AS arm, 'Any E-Mail' AS grp, visit, conversion, spend FROM customers WHERE arm > 0
)
SELECT
    grp                          AS grp,
    arm,
    COUNT(*)                     AS n,
    SUM(visit)                   AS visits,
    SUM(conversion)              AS conversions,
    SUM(spend)                   AS revenue,
    SUM(spend * spend)           AS revenue_sq,
    AVG(spend)                   AS spend_per_customer,
    SUM(spend) / NULLIF(SUM(conversion), 0) AS revenue_per_converter
FROM grouped
GROUP BY grp, arm
ORDER BY arm;
