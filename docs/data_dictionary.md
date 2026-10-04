# Data dictionary

## dim_location (236 rows)
| Column | Meaning |
|---|---|
| location_id | Surrogate key |
| country, location_name | Place as named by the source (usually the capital) |
| latitude, longitude | Degrees |
| timezone | IANA time zone |

## dim_condition (27 rows)
| Column | Meaning |
|---|---|
| condition_id | Surrogate key |
| condition_text | Weather text, lower-cased and trimmed |
| condition_group | `clear / sunny`, `cloudy`, `fog / mist`, `rain / thunder`, `rain nearby / possible` or `other` |
| is_wet | TRUE for rain, drizzle, showers or thunder happening now; FALSE for dry conditions; NULL when the text says rain is only nearby or possible |

## fact_observation (8,720 rows, one per location and observation time)
| Column | Unit | Meaning |
|---|---|---|
| observed_at_local / observed_at_utc | | When the reading was taken |
| temperature_celsius, feels_like_celsius | C | Air temperature and apparent temperature |
| wind_kph, gust_kph, wind_degree | km/h, km/h, degrees | Wind speed, gusts and direction |
| pressure_mb | hPa | Sea-level pressure |
| precip_mm | mm | Precipitation |
| humidity, cloud | % | Relative humidity and cloud cover |
| visibility_km | km | Visibility |
| uv_index | index | UV index (1 is the floor, also overnight) |
| pm25 | ug/m3 | Fine particulate matter |

The raw file also carries Fahrenheit, mph, inch and mile copies of the metric columns. They are dropped after the
unit-consistency checks prove they are exact conversions.

## quarantine_observation (5 rows)
Raw rows removed as sensor errors, with a `reason` column. All five report a sustained wind above 150 km/h (172 to 2,963 km/h)
together with "partly cloudy" and ordinary pressure, so they are treated as faulty readings, not storms.
