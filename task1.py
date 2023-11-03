'''
Redwood eng challenge task 1:
Add a column called corrected_time with best guess of timestamps
'''

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time



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



def select_reference_point(groups):
    # Compare actual Y diff vs Y diff as calculated using the start and end times wrt speed data
    # Get the row with the lowest error and use this as the reference timestamp
    y_errors = [] # Initialize an empty list to store tuples
    for group_key, group in groups:
        ocr_y = group['ocr_y'].iloc[0]
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
            middle_rows = group.iloc[1:]
            middle_moves = (middle_rows['time_diff'] * middle_rows['drum_speed_previous'] / 60).sum()
            drum_speed_Y_travel = first_move + middle_moves + last_move

        # Check the difference in Y travel when calculated using ocr (true) vs speed data from the ocr timestamps (possibly false)
        y_error = group['ocr_y_travel'].iloc[0] - drum_speed_Y_travel
        if y_error and not pd.isna(y_error):
            y_errors.append((ocr_y, abs(y_error), end_time))
    ocr_y, min_y_error, end_time = min(y_errors, key=lambda x: x[1])

    return end_time


def calculate_end_time(step, start_time, travel_distance, speed_changes):
    # Calculates end time for a given y travel distance with a known start time
    remaining_distance = - travel_distance
    current_time = start_time

    for _, row in speed_changes.iterrows():
        # Calculate time to travel the remaining travel_distance at the current speed
        speed_m_per_s = row['drum_speed_previous'] / 60
        time_for_this_segment = remaining_distance / speed_m_per_s

        # Get the time until the next speed change. Total time we can spend at this speed.
        if step == 1:
            time_until_next_speed_change = (row['timestamp'] - current_time).total_seconds()
        else:
            time_until_next_speed_change = (current_time - row['timestamp']).total_seconds()


        if time_for_this_segment <= time_until_next_speed_change:
            # Calculate the exact end_time or start_time based on remaining distance and step
            new_time = current_time + pd.Timedelta(seconds=time_for_this_segment * step)
            return new_time
        else:
            # Travel as much as possible until speed changes
            remaining_distance -= speed_m_per_s * time_until_next_speed_change
            current_time = row['timestamp']

        # If not returned within loop, all remaining distance can be covered with the last speed
        final_time_segment = remaining_distance / speed_m_per_s
        new_time = current_time + pd.Timedelta(seconds=final_time_segment * step)

        return new_time


def fill_timestamps(reference_timestamp, reference_index, step, merged_data, ocr_df_corrected):
    # Corrects the timestamps in ocr_df_corrected using the Y travel from ocr and speeds from speed_df

    previous_time = reference_timestamp
    current_time = 0 # unknown
    index_corrected_df = reference_index + step

    #if step == -1:
        #merged_data = merged_data.sort_values(by='timestamp', ascending=False).copy()

    # Start the fill loop
    while (0 < index_corrected_df < len(ocr_df_corrected)):

        y_travel = ocr_df_corrected.loc[index_corrected_df, 'ocr_y_travel']

        # Get the rest of the speed data that is needed for the fill
        if step == 1:
            speed_changes = merged_data[merged_data['timestamp'] > previous_time]
        else:
            speed_changes = merged_data[merged_data['timestamp'] < previous_time]

        current_time = calculate_end_time(step=step, start_time=previous_time, travel_distance=y_travel, speed_changes=speed_changes)
        #print(index_corrected_df, current_time, y_travel)

        # Update the corrected timestamp in ocr_df_corrected
        ocr_df_corrected.loc[index_corrected_df, 'corrected_timestamp'] = current_time

        # Update the current index_corrected_df and previous_time for ocr_df_corrected
        index_corrected_df += step
        previous_time = current_time  # Set the new time for the next iteration

    return ocr_df_corrected


def main() -> None:
    speed_df = pd.read_csv('./data_task1/speed.csv')
    ocr_df = pd.read_csv('./data_task1/ocr_df.csv', dtype={'ocr_date': str, 'ocr_time': str})

    ocr_df['timestamp'] = ocr_df.apply(create_datetime, axis=1)
    speed_df['timestamp']= pd.to_datetime(speed_df['ts'])


    # Clean up data
    # OCR
    # Used ocr_df_corrected to hold correct timestamps
    ocr_df_corrected = ocr_df.sort_values(by='ocr_y', ascending=False).copy()
    ocr_df_corrected['ocr_y_travel'] = ocr_df_corrected['ocr_y'].diff()
    ocr_df_corrected = ocr_df_corrected.reset_index()
    ocr_df_corrected['corrected_timestamp'] = pd.NaT
    #print(ocr_df_corrected)

    # Used for data analysis
    ocr_df = ocr_df.sort_values(by='timestamp') # Need this to merge with speed data
    ocr_df['ocr_y_travel'] = ocr_df['ocr_y'].diff()*(-1)

    ocr_df['ocr_timestamp_move_end'] = ocr_df['timestamp']
    ocr_df['ocr_timestamp_move_start'] = ocr_df['timestamp'].shift(1) # This is wrong if a timestamp is read wrong, but we check this later on
    ocr_df = ocr_df[['timestamp', 'ocr_timestamp_move_start', 'ocr_timestamp_move_end', 'ocr_y', 'ocr_y_travel']]
    ocr_df = ocr_df.dropna().copy() # Filter out NaT values from ocr_df
    min_speed_timestamp = speed_df['timestamp'].min() # Filter out times outside of speed range
    max_speed_timestamp = speed_df['timestamp'].max()
    ocr_df = ocr_df[(ocr_df['timestamp'] >= min_speed_timestamp) & (ocr_df['timestamp'] <= max_speed_timestamp)]

    # SPEED
    speed_df = speed_df.sort_values(by='timestamp')
    speed_df['drum_speed_next'] = speed_df['drum_speed']
    speed_df['drum_speed_previous'] = speed_df['drum_speed'].shift(1) # Shift drum speeds down by 1 so that drum speed in row stops at that timestamp in that row
    speed_df['time_diff'] = speed_df['timestamp'].diff().dt.total_seconds()
    speed_df = speed_df[['timestamp', 'drum_speed', 'drum_speed_next', 'drum_speed_previous', 'time_diff']]


    # Merge on speed timestamp
    merged_data = pd.merge_asof(speed_df, ocr_df, on='timestamp', direction='forward')
    merged_data = merged_data[['timestamp', 'drum_speed_next', 'drum_speed_previous', 'time_diff','ocr_timestamp_move_start', 'ocr_timestamp_move_end', 'ocr_y', 'ocr_y_travel']] # Reordering


    # Group the merged_data by ocr_timestamp_move_end and get most_accurate_timestamp based on y travel
    groups = merged_data.groupby('ocr_timestamp_move_end')
    most_accurate_timestamp = select_reference_point(groups)
    most_accurate_index = ocr_df_corrected[ocr_df_corrected['timestamp']==most_accurate_timestamp].index.min()

    # Now to fill in timestamps based on the most_accurate_timestamp
    start_time = time.time()
    ocr_df_corrected = fill_timestamps(reference_timestamp=most_accurate_timestamp, reference_index=most_accurate_index, step=1, merged_data=merged_data, ocr_df_corrected=ocr_df_corrected)
    end_time = time.time()
    print('done forward fill in: {}'.format(end_time-start_time))

    ocr_df_corrected.loc[most_accurate_index, 'corrected_timestamp'] = most_accurate_timestamp

    start_time = time.time()
    ocr_df_corrected = fill_timestamps(reference_timestamp=most_accurate_timestamp, reference_index=most_accurate_index, step=-1, merged_data=(merged_data.sort_values(by='timestamp', ascending=False)), ocr_df_corrected=ocr_df_corrected)
    end_time = time.time()
    print('done back fill in: {}'.format(end_time-start_time))



    print('ocr_df_corrected')
    pd.set_option('display.max_rows', None)
    #print(ocr_df_corrected)

    #print(most_accurate_timestamp, most_accurate_index)



    # Plots
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    auto_locator = mdates.AutoDateLocator(maxticks=10)
    # First subplot
    axs[0].plot(
        speed_df["timestamp"],
        speed_df["drum_speed"],
        label="Drum speed",
        color="tab:green"
    )
    axs[0].set_title('Drum speed vs Timestamp')
    axs[0].set_ylabel("Speed (m/min)")
    axs[0].xaxis.set_major_formatter(date_format)
    axs[0].xaxis.set_major_locator(auto_locator)

    # Second subplot
    axs[1].scatter(
        ocr_df_corrected["timestamp"],
        ocr_df_corrected["ocr_y"],
        label="Uncorrected timestamp",
        color="tab:red",
        alpha=0.5
    )
    axs[1].set_title('OCR y vs Timestamp')
    axs[1].set_ylabel("Y distance (m)")
    axs[1].xaxis.set_major_formatter(date_format)
    axs[1].xaxis.set_major_locator(auto_locator)

    # Second subplot
    axs[1].scatter(
        ocr_df_corrected["timestamp"],
        ocr_df_corrected["ocr_y"],
        label="Uncorrected timestamp",
        color="tab:red",
        alpha=0.5
    )
    axs[2].plot(
        ocr_df_corrected["corrected_timestamp"],
        ocr_df_corrected["ocr_y"],
        label="Uncorrected timestamp)",
        color="tab:blue"
    )
    axs[2].set_title('OCR y vs Corrected Timestamp')
    axs[2].set_ylabel("Y distance (m)")
    axs[2].xaxis.set_major_formatter(date_format)
    axs[2].xaxis.set_major_locator(auto_locator)


    axs[-1].set_xlabel("Timestamp")
    plt.show()


if __name__ == "__main__":
    main()
