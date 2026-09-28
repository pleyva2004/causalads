-- Pre-specified segment readout: outcome counts by (dimension, level, arm).
-- Segments are defined only from pre-treatment columns.
WITH long AS (
    SELECT 'channel' AS dimension, channel AS level, arm, visit, conversion FROM customers
    UNION ALL
    SELECT 'zip_code', zip_code, arm, visit, conversion FROM customers
    UNION ALL
    SELECT 'newbie', CASE WHEN newbie = 1 THEN 'new (<12mo)' ELSE 'existing' END, arm, visit, conversion FROM customers
    UNION ALL
    SELECT 'past mens buyer', CASE WHEN mens = 1 THEN 'yes' ELSE 'no' END, arm, visit, conversion FROM customers
    UNION ALL
    SELECT 'past womens buyer', CASE WHEN womens = 1 THEN 'yes' ELSE 'no' END, arm, visit, conversion FROM customers
    UNION ALL
    SELECT 'history bucket', history_segment, arm, visit, conversion FROM customers
)
SELECT dimension, level, arm,
       COUNT(*)        AS n,
       SUM(visit)      AS visits,
       SUM(conversion) AS conversions
FROM long
GROUP BY dimension, level, arm
ORDER BY dimension, level, arm;
