-- Campaign readout by arm: the query an analyst would run first.
SELECT
    segment,
    COUNT(*)                          AS customers,
    AVG(visit)                        AS visit_rate,
    AVG(conversion)                   AS conversion_rate,
    AVG(spend)                        AS spend_per_customer,
    SUM(spend)                        AS revenue
FROM customers
GROUP BY segment
ORDER BY segment;
