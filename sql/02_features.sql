-- Pre-treatment feature engineering. Nothing here may read visit/conversion/spend.
CREATE OR REPLACE TABLE features AS
SELECT
    customer_id,
    recency,
    history,
    LN(1 + history)                                   AS log_history,
    history_bucket,
    -- history is floor-censored at $29.99 for ~12% of customers.
    CAST(history <= 29.99 AS INTEGER)                 AS history_at_floor,
    mens,
    womens,
    CAST(mens = 1 AND womens = 1 AS INTEGER)          AS both_lines,
    newbie,
    CAST(recency <= 3 AS INTEGER)                     AS recent_buyer,
    CAST(zip_code = 'Urban' AS INTEGER)               AS zip_urban,
    CAST(zip_code = 'Suburban' AS INTEGER)            AS zip_suburban,
    CAST(zip_code = 'Rural' AS INTEGER)               AS zip_rural,
    CAST(channel = 'Phone' AS INTEGER)                AS ch_phone,
    CAST(channel = 'Web' AS INTEGER)                  AS ch_web,
    CAST(channel = 'Multichannel' AS INTEGER)         AS ch_multi
FROM customers;
