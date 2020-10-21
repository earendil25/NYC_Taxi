import pandas as pd

PROJECT_ID='mobility-293009'


""" TRAINING SET """
base_query = """
WITH base_data AS 
(
  SELECT nyc_taxi.*, gis.* EXCEPT (zip_code_geom)
  FROM (
    SELECT *
    FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2015`
    WHERE 
        1<= EXTRACT(MONTH from pickup_datetime) and
        EXTRACT(MONTH from pickup_datetime) <= 3
        and pickup_latitude  <= 90 and pickup_latitude >= -90
    ) AS nyc_taxi
  JOIN (
    SELECT zip_code, state_code, state_name, city, county, zip_code_geom
    FROM `bigquery-public-data.geo_us_boundaries.zip_codes`
    WHERE state_code='NY'
    ) AS gis 
  ON ST_CONTAINS(zip_code_geom, st_geogpoint(pickup_longitude, pickup_latitude))
), distinct_datetime AS (
  # Datetime만 distinct해서 가져옴(비어있지 않을것이라 가정)
  SELECT distinct DATETIME_TRUNC(pickup_datetime, hour) as pickup_hour
  FROM base_data
), distinct_zip_code AS (
  # zip_code만 distinct해서 가져옴(이외의 zip_code는 나오지 않을것이라 가정)
  SELECT distinct zip_code
  FROM base_data
), zip_code_datetime_join AS (
  # zip_code와 datetime을 join
  SELECT
    *,
    EXTRACT(MONTH FROM pickup_hour) AS month,
    EXTRACT(DAY FROM pickup_hour) AS day,
    CAST(format_datetime('%u', pickup_hour) AS INT64) -1 AS weekday,
    EXTRACT(HOUR FROM pickup_hour) AS hour
  FROM distinct_zip_code  
  CROSS JOIN distinct_datetime
), agg_data AS (
  # zip_code, datetime별 수요 
  SELECT 
      zip_code,
      DATETIME_TRUNC(pickup_datetime, hour) as pickup_hour,
      COUNT(*) AS cnt
  FROM base_data 
  GROUP BY zip_code, pickup_hour
), join_output AS (
  # zip_code, datetime 데이터에 수요값을 붙이고 없다면 0처리
  select 
    zip_code_datetime.*, 
    IFNULL(agg_data.cnt, 0) AS cnt
  from zip_code_datetime_join as zip_code_datetime
  LEFT JOIN agg_data
  ON zip_code_datetime.zip_code = agg_data.zip_code and zip_code_datetime.pickup_hour = agg_data.pickup_hour
)
SELECT
  *,
  LAG(cnt, 24) OVER(PARTITION BY zip_code ORDER BY pickup_hour) AS lag_1d_cnt,
  LAG(cnt, 168) OVER(PARTITION BY zip_code ORDER BY pickup_hour) AS lag_7d_cnt,
  LAG(cnt, 336) OVER(PARTITION BY zip_code ORDER BY pickup_hour) AS lag_14d_cnt,
  ROUND(AVG(cnt) OVER(PARTITION BY zip_code ORDER BY pickup_hour ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING), 2) AS avg_14d_cnt,
  ROUND(AVG(cnt) OVER(PARTITION BY zip_code ORDER BY pickup_hour ROWS BETWEEN 336 PRECEDING AND 1 PRECEDING), 2) AS avg_21d_cnt,
  CAST(STDDEV(cnt) OVER(PARTITION BY zip_code ORDER BY pickup_hour ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING) AS INT64) AS std_14d_cnt,
  CAST(STDDEV(cnt) OVER(PARTITION BY zip_code ORDER BY pickup_hour ROWS BETWEEN 336 PRECEDING AND 1 PRECEDING) AS INT64) AS std_21d_cnt
FROM join_output
order by zip_code, pickup_hour
"""

climate_query = """
SELECT
  EXTRACT(MONTH from date) AS month,
  EXTRACT(DAY from date) AS day,
  element, value
FROM `bigquery-public-data.ghcn_d.ghcnd_2015`
WHERE id = 'USW00094728' and 
  (element = 'TMAX'or element = 'TMIN' or element = 'PRCP'
   or element = 'SNOW' or element = 'SNWD') and 
   1 <= EXTRACT(MONTH from date) and
   EXTRACT(MONTH from date) <= 3
ORDER BY date"""

base_df = pd.read_gbq(query=base_query, dialect='standard', project_id=PROJECT_ID)
climate_df = pd.read_gbq(query=climate_query, dialect='standard', project_id=PROJECT_ID)

tmax_df = climate_df[climate_df['element']=='TMAX'].reset_index()
tmin_df = climate_df[climate_df['element']=='TMIN'].reset_index()
prcp_df = climate_df[climate_df['element']=='PRCP'].reset_index()
snow_df = climate_df[climate_df['element']=='SNOW'].reset_index()
snwd_df = climate_df[climate_df['element']=='SNWD'].reset_index()
tmax_df['TMAX']=tmax_df['value']
tmax_df['TMIN']=tmin_df['value']
tmax_df['PRCP']=prcp_df['value']
tmax_df['SNOW']=snow_df['value']
tmax_df['SNWD']=snwd_df['value']
del tmax_df['element']
tmax_df.head()


train_data_df = base_df.merge(tmax_df)
train_data_df.to_csv('./train_data_df.csv')

print('Training data set downloaded')

"""TEST SET"""

test_base_query = """
WITH base_data AS 
(
  SELECT nyc_taxi.*, gis.* EXCEPT (zip_code_geom)
  FROM (
    SELECT *
    FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2016`
    WHERE 
        1<=EXTRACT(MONTH from pickup_datetime) and
        EXTRACT(MONTH from pickup_datetime) <=3 and
        
        pickup_latitude  <= 90 and pickup_latitude >= -90
    ) AS nyc_taxi
  JOIN (
    SELECT zip_code, state_code, state_name, city, county, zip_code_geom
    FROM `bigquery-public-data.geo_us_boundaries.zip_codes`
    WHERE state_code='NY'
    ) AS gis 
  ON ST_CONTAINS(zip_code_geom, st_geogpoint(pickup_longitude, pickup_latitude))
), distinct_datetime AS (
  # Datetime만 distinct해서 가져옴(비어있지 않을것이라 가정)
  SELECT distinct DATETIME_TRUNC(pickup_datetime, hour) as pickup_hour
  FROM base_data
), distinct_zip_code AS (
  # zip_code만 distinct해서 가져옴(이외의 zip_code는 나오지 않을것이라 가정)
  SELECT distinct zip_code
  FROM base_data
), zip_code_datetime_join AS (
  # zip_code와 datetime을 join
  SELECT
    *,
    EXTRACT(MONTH FROM pickup_hour) AS month,
    EXTRACT(DAY FROM pickup_hour) AS day,
    CAST(format_datetime('%u', pickup_hour) AS INT64) -1 AS weekday,
    EXTRACT(HOUR FROM pickup_hour) AS hour
  FROM distinct_zip_code  
  CROSS JOIN distinct_datetime
), agg_data AS (
  # zip_code, datetime별 수요 
  SELECT 
      zip_code,
      DATETIME_TRUNC(pickup_datetime, hour) as pickup_hour,
      COUNT(*) AS cnt
  FROM base_data 
  GROUP BY zip_code, pickup_hour
), join_output AS (
  # zip_code, datetime 데이터에 수요값을 붙이고 없다면 0처리
  select 
    zip_code_datetime.*, 
    IFNULL(agg_data.cnt, 0) AS cnt
  from zip_code_datetime_join as zip_code_datetime
  LEFT JOIN agg_data
  ON zip_code_datetime.zip_code = agg_data.zip_code and zip_code_datetime.pickup_hour = agg_data.pickup_hour
)
SELECT
  *,
  LAG(cnt, 24) OVER(PARTITION BY zip_code ORDER BY pickup_hour) AS lag_1d_cnt,
  LAG(cnt, 168) OVER(PARTITION BY zip_code ORDER BY pickup_hour) AS lag_7d_cnt,
  LAG(cnt, 336) OVER(PARTITION BY zip_code ORDER BY pickup_hour) AS lag_14d_cnt,
  ROUND(AVG(cnt) OVER(PARTITION BY zip_code ORDER BY pickup_hour ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING), 2) AS avg_14d_cnt,
  ROUND(AVG(cnt) OVER(PARTITION BY zip_code ORDER BY pickup_hour ROWS BETWEEN 336 PRECEDING AND 1 PRECEDING), 2) AS avg_21d_cnt,
  CAST(STDDEV(cnt) OVER(PARTITION BY zip_code ORDER BY pickup_hour ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING) AS INT64) AS std_14d_cnt,
  CAST(STDDEV(cnt) OVER(PARTITION BY zip_code ORDER BY pickup_hour ROWS BETWEEN 336 PRECEDING AND 1 PRECEDING) AS INT64) AS std_21d_cnt
FROM join_output
order by zip_code, pickup_hour
"""

test_climate_query = """
SELECT
  EXTRACT(MONTH from date) AS month,
  EXTRACT(DAY from date) AS day,
  element, value
FROM `bigquery-public-data.ghcn_d.ghcnd_2016`
WHERE id = 'USW00094728' and 
  (element = 'TMAX'or element = 'TMIN' or element = 'PRCP'
   or element = 'SNOW' or element = 'SNWD') and 
   1<= EXTRACT(MONTH from date) and
   EXTRACT(MONTH from date) <= 3
ORDER BY date"""

base_df = pd.read_gbq(query=test_base_query, dialect='standard', project_id=PROJECT_ID)
climate_df = pd.read_gbq(query=test_climate_query, dialect='standard', project_id=PROJECT_ID)

tmax_df = climate_df[climate_df['element']=='TMAX'].reset_index()
tmin_df = climate_df[climate_df['element']=='TMIN'].reset_index()
prcp_df = climate_df[climate_df['element']=='PRCP'].reset_index()
snow_df = climate_df[climate_df['element']=='SNOW'].reset_index()
snwd_df = climate_df[climate_df['element']=='SNWD'].reset_index()
tmax_df['TMAX']=tmax_df['value']
tmax_df['TMIN']=tmin_df['value']
tmax_df['PRCP']=prcp_df['value']
tmax_df['SNOW']=snow_df['value']
tmax_df['SNWD']=snwd_df['value']
del tmax_df['element']
tmax_df.head()


test_data_df = base_df.merge(tmax_df)
test_data_df.to_csv('./test_data_df.csv')

print('Test data set downloaded')