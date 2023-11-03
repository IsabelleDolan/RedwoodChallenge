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
    if pd.isnull(date) or pd.isnull(time):
        return pd.NaT
    timestamp_str = date + time
    try:
        timestamp_datetime = datetime.strptime((timestamp_str), '%y%m%d%H:%M:%S')
    except ValueError as e:
        print(e)
    return timestamp_datetime




ocr_df['timestamp'] = ocr_df.apply(create_datetime, axis=1)
speed_df['timestamp']= pd.to_datetime(speed_df['ts'])


# Clean up data
# OCR
ocr_df = ocr_df.sort_values(by='ocr_y', ascending=False)
ocr_df['ocr_y_travel_true'] = ocr_df['ocr_y'].diff() # Important to do this before sorting by timestamp
ocr_df = ocr_df.sort_values(by='timestamp') # Need this to merge with speed data
ocr_df['ocr_y_travel'] = ocr_df['ocr_y'].diff()*(-1)

ocr_df['ocr_timestamp_move_end'] = ocr_df['timestamp']
ocr_df['ocr_timestamp_move_start'] = ocr_df['timestamp'].shift(1) # This is wrong if a timestamp is read wrong, but we check this later on
ocr_df['ocr_y_at_move_end'] = ocr_df['ocr_y']
ocr_df = ocr_df[['timestamp', 'ocr_timestamp_move_start', 'ocr_timestamp_move_end', 'ocr_y_at_move_end', 'ocr_y_travel', 'ocr_y_travel_true']]
ocr_df_clean = ocr_df.dropna().copy() # Filter out NaT values from ocr_df
min_speed_timestamp = speed_df['timestamp'].min() # Filter out times outside of speed range
max_speed_timestamp = speed_df['timestamp'].max()
ocr_df_clean = ocr_df_clean[(ocr_df_clean['timestamp'] >= min_speed_timestamp) &
                            (ocr_df_clean['timestamp'] <= max_speed_timestamp)]

# SPEED
speed_df = speed_df.sort_values(by='timestamp')
speed_df['drum_speed_next'] = speed_df['drum_speed']
speed_df['drum_speed_previous'] = speed_df['drum_speed'].shift(1) # Shift drum speeds down by 1 so that drum speed in row stops at that timestamp in that row
speed_df = speed_df[['timestamp', 'drum_speed_next', 'drum_speed_previous']]
speed_df['time_diff'] = speed_df['timestamp'].diff().dt.total_seconds()
time_max = ocr_df['timestamp'].max() #Filter out times outside of ocr range to speed things up
time_min = ocr_df['timestamp'].min()
speed_df['timestamp'] = speed_df['timestamp'].mask(~speed_df['timestamp'].between(time_min, time_max))
speed_df = speed_df.dropna().copy()

# Merge on speed timestamp
merged_data = pd.merge_asof(speed_df, ocr_df_clean, on='timestamp', direction='forward')
merged_data = merged_data[['timestamp', 'drum_speed_next', 'drum_speed_previous', 'time_diff','ocr_timestamp_move_start', 'ocr_timestamp_move_end', 'ocr_y_at_move_end', 'ocr_y_travel']] # Reordering



# Compare actually Y diff vs Y diff as calculated using the start and end times wrt speed data
# Flag true/false timestamps in correct_timestamp

# Group the merged_data by ocr_y_at_move_end
groups = merged_data.groupby('ocr_timestamp_move_end')
acceptable_error = 1 # Meters
# Make new column to keep track of correct timestamps
merged_data['correct_timestamp'] = False

# Loop through the groups
for group_key, group in groups:
    start_time = group['ocr_timestamp_move_start'].iloc[0]
    end_time = group['ocr_timestamp_move_end'].iloc[0]
    start_time_diff = (group['timestamp'].iloc[0] - start_time).total_seconds()
    end_time_diff = (end_time - group['timestamp'].iloc[-1]).total_seconds()

    # Calculate Y travel based on the drum speeds
    first_move = start_time_diff*group['drum_speed_previous'].iloc[0]/60
    last_move = end_time_diff*group['drum_speed_next'].iloc[-1]/60
    drum_speed_Y_travel = first_move + last_move
    # If longer than 1 row, need to account for middle rows
    if len(group) > 1:
        # Calculate the sum of the Y travel for the middle rows (ie. all but the first)
        middle_rows = group.iloc[1:]
        middle_moves = (middle_rows['time_diff'] * middle_rows['drum_speed_previous'] / 60).sum()
        drum_speed_Y_travel = first_move + middle_moves + last_move

    # Check the difference in Y travel when calculated using ocr (true) vs speed data from timestamps (possibly false)
    Y_error = group['ocr_y_travel'].iloc[0] - drum_speed_Y_travel

    # Determine if the difference is within the acceptable error margin and update 'correct_timestamp' column
    is_correct = abs(Y_error) < acceptable_error

    # Update the 'correct_timestamp' column for the current group
    merged_data.loc[group.index, 'correct_timestamp'] = is_correct

# Curently stuck with the merged_data correct_timestamp not showing the is_correct properly


    print('GROUP')
    print(merged_data.loc[group.index, 'correct_timestamp'])

    print(start_time_diff, end_time_diff, drum_speed_Y_travel, Y_error, abs(Y_error), acceptable_error, is_correct, group.index)
    print(group)



print('ocr_df_clean')
pd.set_option('display.max_rows', None)
#print(ocr_df)
print('SPEED')
pd.set_option('display.max_rows', 10)
#print(speed_df)
print('MERGED')
#
pd.set_option('display.max_rows', None)
#print(merged_data)




'''


XXX REMOVE XXX
# Disregard ocr timestamps that are outside of speed timestamp range
time_max = speed_df['timestamp'].max()
time_min = speed_df['timestamp'].min()
ocr_df['timestamp'] = ocr_df['timestamp'].mask(~ocr_df['timestamp'].between(time_min, time_max)) # Where True -> replaced with NaT, so have to do complement
XXX REMOVE XXX

XXX REMOVE XXX
# Option A
# ocr - mark all rows with non null timestamps
#non_null_rows = ocr_df['timestamp'].notnull()
# ocr - get Y travel between non null timestamps from ocr_y
#ocr_df.loc[non_null_rows, 'y_travel_ocr'] = ocr_df.loc[non_null_rows, 'ocr_y'].diff().abs()
XXX REMOVE XXX

XXX REMOVE XXX
# ocr - make new df with all the non null timestamps
#ocr_non_null_df = ocr_df.dropna().copy()
# ocr - get Y travel between these non null timestamps from ocr_y
#ocr_non_null_df['ocr_y_travel'] = ocr_non_null_df['ocr_y'].diff()*(-1)
XXX REMOVE XXX

XXX REMOVE XXX
# speed - get Y travel between timestamps in speed
speed_df['time_diff'] = speed_df['timestamp'].diff().dt.total_seconds()
speed_df['drum_speed_y_travel'] = (speed_df['drum_speed'] / 60) * speed_df['time_diff'].shift(-1) # Need to shift time_diff back by 1 to line up with drum_speed
XXX REMOVE XXX


XXX REMOVE XXX
# REMOVE THIS LATER
# First remove speed data outside of ocr data. Not necessary but makes it easier to read when I print it out
time_max = ocr_df['timestamp'].max()
time_min = ocr_df['timestamp'].min()
speed_df['timestamp'] = speed_df['timestamp'].mask(~speed_df['timestamp'].between(time_min, time_max))
speed_df = speed_df.dropna().copy()
XXX REMOVE XXX



# Group the merged_data by ocr_y_at_move_end
groups = merged_data.groupby('ocr_timestamp_move_end')
acceptable_error = 0.1 # Meters
# Make new column to keep track of correct timestamps
merged_data['correct_timestamp'] = False
correct_rows = []

# Loop through the groups
for group_key, group in groups:
    start_time = group['ocr_timestamp_move_start'].iloc[0]
    end_time = group['ocr_timestamp_move_end'].iloc[0]
    start_time_diff = (group['timestamp'].iloc[0] - start_time).total_seconds()
    end_time_diff = (end_time - group['timestamp'].iloc[-1]).total_seconds()

    # Calculate Y travel based on the drum speeds
    first_move = start_time_diff*group['drum_speed_previous'].iloc[0]/60
    last_move = end_time_diff*group['drum_speed_next'].iloc[-1]/60
    drum_speed_Y_travel = first_move + last_move
    # If longer than 1 row, need to account for middle rows
    if len(group) > 1:
        # Calculate the sum of the Y travel for the middle rows (ie. all but the first)
        middle_rows = group.iloc[1:]
        middle_moves = (middle_rows['time_diff'] * middle_rows['drum_speed_next'] / 60).sum()
        drum_speed_Y_travel = first_move + middle_moves + last_move

    # Check the difference in Y travel when calculated using ocr (true) vs speed data from timestamps (possibly false)
    Y_error = group['ocr_y_travel'].iloc[0] - drum_speed_Y_travel

    # Determine if the difference is within the acceptable error margin and update 'correct_timestamp' column
    is_correct = abs(Y_error) < acceptable_error

    # Update the 'correct_timestamp' column for the current group
    merged_data.loc[group.index, 'correct_timestamp'] = is_correct




# Filling in the incorrect timestamps
# Reorder based on ocr Y so we can go in descending rows. This is needed so that the end_time and become the start time for the next one.


def calculate_end_time(start_time, travel_distance, speed_changes):
    # Calculates end time for a given y travel distance with a known start time
    remaining_distance = travel_distance
    current_time = start_time

    for _, row in speed_changes.iterrows():
        # Calculate time to travel the remaining travel_distance at the current speed
        speed_m_per_s = row['drum_speed_previous'] / 60
        time_for_this_segment = remaining_distance / speed_m_per_s

        # Get the time until the next speed change. Total time we can spend at this speed.
        time_until_next_speed_change = (row['timestamp'] - current_time).total_seconds()

        if time_for_this_segment > time_until_next_speed_change:
            # If the time it takes to complete the y travel at this speed is greater than the time to the next speed change,
            # travel as much as possible until the speed changes
            remaining_distance -= speed_m_per_s * time_until_next_speed_change
            current_time = row['timestamp']
        else:
            # If the remaining distance can be traveled before the next speed change, calculate the exact end_time and break from the loop
            end_time = current_time + pd.Timedelta(seconds=time_for_this_segment)
            return end_time



# Find the first correct timestamp
first_correct_index = merged_data[merged_data['correct_timestamp']].index.min()
first_correct_timestamp = merged_data.loc[first_correct_index, 'timestamp']
previous_y_move = None
start_times = []
end_times = []
total_y_travels = []
y_at_move_ends = []

#Filling forward
for index in range(first_correct_index + 1, len(merged_data)):
    current_y_move = merged_data.loc[index, 'ocr_y_at_move_end']
    # If the timestamp is incorrect, calculate the correct one. Make sure we're on a new y_move
    if not merged_data.loc[index, 'correct_timestamp'] and current_y_move != previous_y_move:
        total_y_travel = merged_data.loc[index, 'ocr_y_travel']
        start_time = merged_data.loc[index, 'ocr_timestamp_move_start']
        speed_changes_after = merged_data[merged_data['timestamp'] > merged_data.loc[index, 'timestamp']]
        ocr_timestamp_move_end = calculate_end_time(start_time, total_y_travel, speed_changes_after)

        # Add the data to the lists
        start_times.append(start_time)
        end_times.append(ocr_timestamp_move_end)
        total_y_travels.append(total_y_travel)
        y_at_move_ends.append(merged_data.loc[index, 'ocr_y_at_move_end'])

        # Update the merged_data with the new end time
        merged_data.loc[index, 'ocr_timestamp_move_end'] = ocr_timestamp_move_end

        # Update the previous_y_move
        previous_y_move = current_y_move


# Create a dictionary from the lists
data_dict = {
    'start_time': start_times,
    'end_time': end_times,
    'total_y_travel': total_y_travels,
    'y_at_move_end': y_at_move_ends
}

# Convert the dictionary to a pandas DataFrame
corrected_timestamps_df = pd.DataFrame(data_dict)
print(corrected_timestamps_df)
'''


# Look from first correct timestamp to end of merge_data

# If correct_timestamp is False, look at ocr_y_travel, ocr_timestamp_move_start, drum_speed_previous, and timestamp

# We were at previous speed from ocr_timestamp_move_start to timestamp
# total_y_travel = ocr_y_travel
# sum_y_travel = (merged_data['ocr_timestamp_move_start'].iloc[index] - merged_data['timestamp'].iloc[index]).total_seconds
# if sum_y_travel >= total_y_travel:
    # ocr_timestamp_move_end  = timestamp
    # merged_data.loc[group.index, 'ocr_timestamp_move_end'] = ocr_timestamp_move_end
# If not, go to next index and add time_diff*drum_speed_next/60
# Check after each one if we are above













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





