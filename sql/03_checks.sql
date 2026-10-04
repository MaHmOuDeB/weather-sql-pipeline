-- Data-quality checks. Each row: name, severity, number of failing rows.  Severity 'error' stops the pipeline.
SELECT 'staging row count equals fact row count' AS check_name, 'error' AS severity,
       ABS((SELECT COUNT(*) FROM stg_weather) - (SELECT COUNT(*) FROM fact_observation)) AS failing_rows
UNION ALL SELECT 'location + observation time is unique', 'error',
       (SELECT COUNT(*) FROM (SELECT location_id, observed_at_utc FROM fact_observation GROUP BY ALL HAVING COUNT(*) > 1))
UNION ALL SELECT 'no missing keys or timestamps', 'error',
       (SELECT COUNT(*) FROM fact_observation WHERE location_id IS NULL OR condition_id IS NULL OR observed_at_utc IS NULL)
UNION ALL SELECT 'humidity within 0-100', 'error', (SELECT COUNT(*) FROM fact_observation WHERE humidity NOT BETWEEN 0 AND 100)
UNION ALL SELECT 'cloud cover within 0-100', 'error', (SELECT COUNT(*) FROM fact_observation WHERE cloud NOT BETWEEN 0 AND 100)
UNION ALL SELECT 'UV index not negative', 'error', (SELECT COUNT(*) FROM fact_observation WHERE uv_index < 0)
UNION ALL SELECT 'precipitation and visibility not negative', 'error', (SELECT COUNT(*) FROM fact_observation WHERE precip_mm < 0 OR visibility_km < 0)
UNION ALL SELECT 'pressure plausible (850-1090 hPa)', 'error', (SELECT COUNT(*) FROM fact_observation WHERE pressure_mb NOT BETWEEN 850 AND 1090)
UNION ALL SELECT 'temperature plausible (-60 to 60 C)', 'error', (SELECT COUNT(*) FROM fact_observation WHERE temperature_celsius NOT BETWEEN -60 AND 60)
UNION ALL SELECT 'latitude / longitude valid', 'error', (SELECT COUNT(*) FROM dim_location WHERE latitude NOT BETWEEN -90 AND 90 OR longitude NOT BETWEEN -180 AND 180)
UNION ALL SELECT 'Celsius and Fahrenheit columns agree (within 0.5 F)', 'error',
       (SELECT COUNT(*) FROM stg_weather WHERE ABS(temperature_celsius * 9 / 5 + 32 - temperature_fahrenheit) > 0.5)
UNION ALL SELECT 'kph and mph columns agree (within 0.2 mph)', 'error',
       (SELECT COUNT(*) FROM stg_weather WHERE ABS(wind_kph / 1.609344 - wind_mph) > 0.2)
UNION ALL SELECT 'gusts not lower than sustained wind', 'warn', (SELECT COUNT(*) FROM fact_observation WHERE gust_kph < wind_kph)
UNION ALL SELECT 'wind above 150 km/h (review, may be real)', 'warn', (SELECT COUNT(*) FROM fact_observation WHERE wind_kph > 150)
UNION ALL SELECT 'condition spellings merged by lower-casing and trimming', 'info',
       (SELECT COUNT(DISTINCT condition_text) - COUNT(DISTINCT lower(trim(condition_text))) FROM read_csv_auto('{csv_path}'))
UNION ALL SELECT 'locations observed fewer than 20 times', 'warn',
       (SELECT COUNT(*) FROM (SELECT location_id FROM fact_observation GROUP BY location_id HAVING COUNT(*) < 20));
