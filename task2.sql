/* 
Written with Postgres version 16 in pgAdmin 4 
*/

-- CTE that contains timestamps in 1 min intervals from min time to max time
WITH RECURSIVE minutes_table AS (
  -- Getting the minimum timestmap, rounded down to the minute
  SELECT DATE_TRUNC('minute', min(timestamp)) AS dt FROM tag_data
  UNION ALL
  -- Combine the minimum timestamp with the rest of the 1 minute incremented timestamps  
  SELECT dt + INTERVAL '1 minute' FROM minutes_table WHERE dt < (SELECT max(timestamp) FROM tag_data)
),
-- CTE with column called next_timestamp to make calculaitons easier
-- UPDATE: This is not a cte! 
last_values AS (
  SELECT 
    tagid, 
    float_value, 
    timestamp, 
	-- Lead looks at next timestamp for each tagid, ordered by timestamp
    LEAD(timestamp) OVER (PARTITION BY tagid ORDER BY timestamp) AS next_timestamp
  FROM tag_data
),
-- CTE for weighted sensor values 
time_weighted_values AS (
  SELECT 
    last_values.tagid,
    m_table.dt AS minute, -- getting times from minute table 
	-- EXTRACT(epoch FROM is get total num of seconds (same flavour as .total_seconds())
	-- Seeing which is less, doing the full minute or going to the next timestamp. The next timestamp might be null, hence the COALESCE
	-- Seeing which is more, the last timestamp or minute timestamp. We need the greater one so we stay within the 1 minute window. 
    SUM(last_values.float_value * EXTRACT(epoch FROM LEAST(m_table.dt + interval '1 minute', last_values.next_timestamp) - GREATEST(last_values.timestamp, m_table.dt))) -- this is weighted avg
    / NULLIF(EXTRACT(epoch FROM SUM(LEAST(m_table.dt + interval '1 minute', last_values.next_timestamp) - GREATEST(last_values.timestamp, m_table.dt))), 0) AS time_weighted_avg  --dividing by total num seconds coevred by readings
  FROM 
    last_values 
  JOIN 
    minutes_table m_table ON last_values.timestamp < m_table.dt + interval '1 minute' AND (last_values.next_timestamp IS NULL OR last_values.next_timestamp > m_table.dt)
  GROUP BY 
    last_values.tagid, m_table.dt, last_values.float_value
)
-- Main
SELECT 
  tagid,
  minute AS timestamp,
  time_weighted_avg
FROM 
  time_weighted_values
ORDER BY 
  -- Must order by tag id first then timestamp
  tagid, 
  timestamp;
