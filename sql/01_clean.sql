-- Clean and type the raw Hillstrom extract. `raw` is registered from pandas with a
-- stable customer_id (row order of the source file).
CREATE OR REPLACE TABLE customers AS
SELECT
    customer_id,
    CAST(recency AS INTEGER)                                   AS recency,
    CAST(history AS DOUBLE)                                    AS history,
    TRIM(history_segment)                                      AS history_segment,
    CAST(regexp_extract(history_segment, '^(\d+)\)', 1) AS INTEGER) AS history_bucket,
    CAST(mens AS INTEGER)                                      AS mens,
    CAST(womens AS INTEGER)                                    AS womens,
    -- Source file misspells 'Suburban' as 'Surburban' (28,776 rows).
    CASE TRIM(zip_code) WHEN 'Surburban' THEN 'Suburban' ELSE TRIM(zip_code) END AS zip_code,
    CAST(newbie AS INTEGER)                                    AS newbie,
    TRIM(channel)                                              AS channel,
    TRIM(segment)                                              AS segment,
    CASE TRIM(segment)
        WHEN 'No E-Mail'     THEN 0
        WHEN 'Mens E-Mail'   THEN 1
        WHEN 'Womens E-Mail' THEN 2
    END                                                        AS arm,
    CAST(TRIM(segment) <> 'No E-Mail' AS INTEGER)              AS treated,
    CAST(visit AS INTEGER)                                     AS visit,
    CAST(conversion AS INTEGER)                                AS conversion,
    CAST(spend AS DOUBLE)                                      AS spend
FROM raw;
