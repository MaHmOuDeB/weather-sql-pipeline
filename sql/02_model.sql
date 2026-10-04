-- Star schema: dim_location, dim_condition, fact_observation.
CREATE OR REPLACE TABLE dim_location AS
SELECT ROW_NUMBER() OVER (ORDER BY country, location_name) AS location_id,
       country, location_name, ANY_VALUE(latitude) AS latitude, ANY_VALUE(longitude) AS longitude, ANY_VALUE(timezone) AS timezone
FROM stg_weather GROUP BY country, location_name;

-- Conditions are lower-cased in staging, so 'Partly Cloudy' and 'Partly cloudy' are one value.
-- is_wet: rain, drizzle, showers or thunder happening now.  'nearby' / 'possible' are ambiguous, so they are NULL (left out of the wet/dry target).
CREATE OR REPLACE TABLE dim_condition AS
SELECT ROW_NUMBER() OVER (ORDER BY condition_clean) AS condition_id, condition_clean AS condition_text,
       CASE WHEN regexp_matches(condition_clean, 'fog|mist')                         THEN 'fog / mist'
            WHEN regexp_matches(condition_clean, 'rain|drizzle|shower|thunder')
                 AND regexp_matches(condition_clean, 'nearby|possible')              THEN 'rain nearby / possible'
            WHEN regexp_matches(condition_clean, 'rain|drizzle|shower|thunder')      THEN 'rain / thunder'
            WHEN regexp_matches(condition_clean, 'overcast|cloudy')                  THEN 'cloudy'
            WHEN regexp_matches(condition_clean, 'sunny|clear')                      THEN 'clear / sunny'
            ELSE 'other' END AS condition_group,
       CASE WHEN regexp_matches(condition_clean, 'nearby|possible')                  THEN NULL
            WHEN regexp_matches(condition_clean, 'rain|drizzle|shower|thunder')      THEN TRUE
            ELSE FALSE END AS is_wet
FROM (SELECT DISTINCT condition_clean FROM stg_weather);

-- Sensor errors are quarantined, not silently kept: all five readings above 150 km/h report 'partly cloudy' at ordinary
-- pressure, and one says 2,963 km/h.
CREATE OR REPLACE TABLE quarantine_observation AS
SELECT s.*, 'wind above 150 km/h with ordinary conditions' AS reason FROM stg_weather s WHERE s.wind_kph > 150;

CREATE OR REPLACE TABLE fact_observation AS
SELECT l.location_id, c.condition_id, s.observed_at_local, s.observed_at_utc,
       s.temperature_celsius, s.feels_like_celsius, s.wind_kph, s.wind_degree, s.gust_kph, s.pressure_mb, s.precip_mm,
       s.humidity, s.cloud, s.visibility_km, s.uv_index, s.pm25
FROM stg_weather s
JOIN dim_location  l ON l.country = s.country AND l.location_name = s.location_name
JOIN dim_condition c ON c.condition_text = s.condition_clean
WHERE NOT COALESCE(s.wind_kph > 150, FALSE);
