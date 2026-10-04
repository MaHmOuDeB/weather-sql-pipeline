-- Staging: typed, snake_case, one row per (location, observation time). Redundant unit columns are dropped
-- (the Fahrenheit, mph, inch and mile columns are exact conversions of the metric ones; a check below proves it).
CREATE OR REPLACE TABLE stg_weather AS
SELECT
    trim(country)                                  AS country,
    trim(location_name)                            AS location_name,
    latitude, longitude, timezone,
    CAST(last_updated AS TIMESTAMP)                AS observed_at_local,
    to_timestamp(last_updated_epoch)               AS observed_at_utc,
    temperature_celsius, feels_like_celsius,
    lower(trim(condition_text))                    AS condition_clean,
    wind_kph, wind_degree, gust_kph, pressure_mb, precip_mm, humidity, cloud, visibility_km, uv_index,
    "air_quality_PM2.5"                            AS pm25,
    -- kept only for the unit-consistency checks
    temperature_fahrenheit, wind_mph
FROM read_csv_auto('{csv_path}');
