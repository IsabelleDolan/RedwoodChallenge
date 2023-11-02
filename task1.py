'''
Redwood eng challenge task 1:
Add a column called corrected_time with best guess of timestamps
'''


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



speed_df = pd.read_csv('./data_task1/speed.csv')
ocr_df = pd.read_csv('./data_task1/ocr_df.csv', dtype={'ocr_date': str, 'ocr_time': str})


# Convert timestamps str to datetime in ocr and speed data
# Turn date + time into a timestamp so it matches speed.csv format, handle empty cells so they get NaT for the resulting timestamp
def create_datetime(row):
    date = row['ocr_date'] # YYMMDD
    time = row['ocr_time'] # HH:MM:SS
    #Deal with empty cells
    if pd.isnull(date) or pd.isnull(time):
        return np.nan
    timestamp_str = date + time
    try:
        timestamp_datetime = datetime.strptime((timestamp_str), '%y%m%d%H:%M:%S')
    except ValueError as e:
        print(e)
    return timestamp_datetime

ocr_df['timestamp'] = ocr_df.apply(create_datetime, axis=1)
speed_df['timestamp']= pd.to_datetime(speed_df['ts'])


# Clean up data
ocr_df = ocr_df.sort_values(by='timestamp')
ocr_df['ocr_timestamp_move_end'] = ocr_df['timestamp']
ocr_df['ocr_timestamp_move_start'] = ocr_df['timestamp'].shift(1)
ocr_df['ocr_y_at_move_end'] = ocr_df['ocr_y']
speed_df = speed_df.sort_values(by='timestamp')
ocr_df.drop(['Unnamed: 0','ocr_date'], axis=1, inplace=True)
speed_df.drop(['Unnamed: 0', 'ts'], axis=1, inplace=True)


# Disregard ocr timestamps that are outside of speed timestamp range
time_max = speed_df['timestamp'].max()
time_min = speed_df['timestamp'].min()
ocr_df['timestamp'] = ocr_df['timestamp'].mask(~ocr_df['timestamp'].between(time_min, time_max)) # Where True -> replaced with NaT, so have to do complement

# Option A
# ocr - mark all rows with non null timestamps
#non_null_rows = ocr_df['timestamp'].notnull()
# ocr - get Y travel between non null timestamps from ocr_y
#ocr_df.loc[non_null_rows, 'y_travel_ocr'] = ocr_df.loc[non_null_rows, 'ocr_y'].diff().abs()

# Option B
# ocr - make new df with all the non null timestamps
ocr_non_null_df = ocr_df.dropna().copy()
# ocr - get Y travel between these non null timestamps from ocr_y
ocr_non_null_df['ocr_y_move_travel'] = ocr_non_null_df['ocr_y'].diff()*(-1)



# speed - get Y travel between timestamps in speed
speed_df['time_diff'] = speed_df['timestamp'].diff().dt.total_seconds()
speed_df['drum_speed_y_travel'] = (speed_df['drum_speed'] / 60) * speed_df['time_diff'].shift(-1) # Need to shift time_diff back by 1 to line up with drum_speed

# Shift drum speeds down by 1 so that drum speed in row stops at that timestamp in that row
speed_df['drum_speed_after_this_timestamp'] = speed_df['drum_speed']
speed_df['drum_speed_before_this_timestamp'] = speed_df['drum_speed'].shift(1)

# First remove speed data outside of ocr data. Not necessary but makes it easier to read when I print it out
time_max = ocr_df['timestamp'].max()
time_min = ocr_df['timestamp'].min()
speed_df['timestamp'] = speed_df['timestamp'].mask(~speed_df['timestamp'].between(time_min, time_max))
speed_df = speed_df.dropna().copy()


# Merge asof on speed timestamp
merged_data = pd.merge_asof(speed_df, ocr_non_null_df, on='timestamp', direction='forward')
merged_data = merged_data[['timestamp', 'drum_speed_after_this_timestamp', 'drum_speed_before_this_timestamp', 'time_diff', 'drum_speed_y_travel', 'ocr_timestamp_move_start', 'ocr_timestamp_move_end', 'ocr_y_at_move_end', 'ocr_y_move_travel']] # Ordering


# Group the merged_data by ocr_y_at_move_end
groups = merged_data.groupby('ocr_y_at_move_end')

# Loop through the groups
for y_at_move_end, group in groups:
    start_time = group['ocr_timestamp_move_start']
    end_time = group['ocr_timestamp_move_end']
    start_time_diff = (group['timestamp'].iloc[0] - start_time).total_seconds()
    end_time_diff = (end_time - group['timestamp'].iloc[-1]).total_seconds()

    #Sum middle times. keep array of caculated Y? can als compare directly

# Merge speed and orc
#merged_data = pd.merge_asof(speed_df.dropna(), ocr_non_null_df, on='timestamp')
#merged_data = merged_data[['timestamp', 'drum_speed', 'time_diff', 'drum_speed_y_travel', 'ocr_timestamp', 'ocr_y', 'ocr_y_travel']] # Reorganizing


# speed - get closest equal to or less than timestamps from speed. Calculate Y travel from times and speeds
# For timestamp[i] and timestamp[i-1], Y travel = y_diff[i]
# In speed, Y travel = speed[i-1]*(timestamp[i] - timestamp[i-1])

# Compare Y's. If good, add new column to ocr with True beside good timestamps


'''
#for i in range(len(ocr_non_null_df)):
for i in range(4):
    start_time = ocr_non_null_df['timestamp'].iloc[i]
    end_time = ocr_non_null_df['timestamp'].iloc[i+1]
    ocr_y_travel = ocr_non_null_df['ocr_y_travel'].iloc[i]

    drum_speed_segments = speed_df[(speed_df['timestamp']>=start_time)&(speed_df['timestamp']<=end_time)]
    print('start, end')
    print(start_time, end_time)
    print(drum_speed_segments)
'''





print('OCR')
pd.set_option('display.max_rows', None)
#print(ocr_df)
print('SPEED')
pd.set_option('display.max_rows', 10)
#print(speed_df)
print('ocr_non_null_df')
pd.set_option('display.max_rows', None)
#print(ocr_non_null_df)

print('MERGED')
#pd.set_option('display.max_rows', None)
print(merged_data)


'''
# Plots
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
auto_locator = mdates.AutoDateLocator(maxticks=10)
# First subplot
axs[0].scatter(
    speed_df["timestamp"],
    speed_df["drum_speed"],
    label="Drum speed",
    color="tab:green",
    alpha=0.5
)
axs[0].set_title('Drum speed vs Timestamp')
axs[0].set_ylabel("Speed (m/min)")
axs[0].xaxis.set_major_formatter(date_format)
axs[0].xaxis.set_major_locator(auto_locator)

# Second subplot
axs[1].scatter(
    ocr_df["timestamp"],
    ocr_df["ocr_y"],
    label="y as read by ocr (m)",
    color="tab:blue",
    alpha=0.5
)
axs[1].set_title('OCR y vs Timestamp')
axs[1].set_ylabel("Distance from 0 (m)")
axs[1].xaxis.set_major_formatter(date_format)
axs[1].xaxis.set_major_locator(auto_locator)

# Third subplot
axs[2].scatter(
    ocr_df["timestamp"],
    ocr_df["y_diff"],
    label="y_diff",
    color="tab:red",
    alpha=0.5
)
axs[2].set_title('y_diff')
axs[2].set_ylabel("Y (m)")
axs[2].xaxis.set_major_formatter(date_format)
axs[2].xaxis.set_major_locator(auto_locator)

axs[-1].set_xlabel("Timestamp")
#plt.show()

'''





