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

# ocr - make new df with all the non null timestamps
ocr_non_null_df = ocr_df.dropna().copy()
# ocr - get Y travel between these non null timestamps from ocr_y
ocr_non_null_df['ocr_y_move_travel'] = ocr_non_null_df['ocr_y'].diff()*(-1)


# speed - get Y travel between timestamps in speed
speed_df['time_diff'] = speed_df['timestamp'].diff().dt.total_seconds()
speed_df['drum_speed_y_travel'] = (speed_df['drum_speed'] / 60) * speed_df['time_diff'].shift(-1) # Need to shift time_diff back by 1 to line up with drum_speed

# Shift drum speeds down by 1 so that drum speed in row stops at that timestamp in that row
speed_df['drum_speed_next'] = speed_df['drum_speed']
speed_df['drum_speed_previous'] = speed_df['drum_speed'].shift(1)

# REMOVE THIS LATER
# First remove speed data outside of ocr data. Not necessary but makes it easier to read when I print it out
time_max = ocr_df['timestamp'].max()
time_min = ocr_df['timestamp'].min()
speed_df['timestamp'] = speed_df['timestamp'].mask(~speed_df['timestamp'].between(time_min, time_max))
speed_df = speed_df.dropna().copy()


# Merge asof on speed timestamp
merged_data = pd.merge_asof(speed_df, ocr_non_null_df, on='timestamp', direction='forward')
merged_data = merged_data[['timestamp', 'drum_speed_next', 'drum_speed_previous', 'time_diff', 'drum_speed_y_travel', 'ocr_timestamp_move_start', 'ocr_timestamp_move_end', 'ocr_y_at_move_end', 'ocr_y_move_travel']] # Ordering


# Group the merged_data by ocr_y_at_move_end
groups = merged_data.groupby('ocr_timestamp_move_end')

errors = []

# Loop through the groups
for group_key, group in groups:
    start_time = group['ocr_timestamp_move_start'].iloc[0]
    end_time = group['ocr_timestamp_move_end'].iloc[0]
    start_time_diff = (group['timestamp'].iloc[0] - start_time).total_seconds()
    end_time_diff = (end_time - group['timestamp'].iloc[-1]).total_seconds()

    # Calculate Y travel based on the drum speeds
    first_move = start_time_diff*group['drum_speed_previous'].iloc[0]/60
    first_move_false = (start_time_diff+1)*group['drum_speed_previous'].iloc[0]/60 # delete
    last_move = end_time_diff*group['drum_speed_next'].iloc[-1]/60
    drum_speed_Y_travel = first_move + last_move
    drum_speed_Y_travel_off_by_1 = first_move_false + last_move # delete

    # If longer than 1 row, need to account for middle rows
    if len(group) > 1:
        # Calculate the sum of the Y travel for the middle rows (all but the first)
        middle_rows = group.iloc[1:]
        middle_moves = (middle_rows['time_diff'] * middle_rows['drum_speed_next'] / 60).sum()
        drum_speed_Y_travel = first_move + middle_moves + last_move

        drum_speed_Y_travel_off_by_1 = first_move_false + middle_moves + last_move # delete

    speed_ocr_Y_diff = group['ocr_y_move_travel'].iloc[0] - drum_speed_Y_travel

    error = round((drum_speed_Y_travel_off_by_1 - speed_ocr_Y_diff), 4)


    if -1 < speed_ocr_Y_diff < 1:
        errors.append(abs(error))

        #print('speed_ocr_Y_diff')
        print('-------------')
        print(round(speed_ocr_Y_diff,4))
        print(round(drum_speed_Y_travel_off_by_1,4))
        print(error)

print('Overall min and max tolerances')
print(min(errors))
print(max(errors))



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
