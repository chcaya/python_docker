import subprocess

import pandas as pd
import numpy as np

from ardupilot_log_reader.reader import Ardupilot
from pathlib import Path

import bisect

from pyproj import Geod
from affine import Affine


def extract_rising_edges(_timestamps, _signal, _threshold):
    rising_edges_timestamps = []
    is_rised = False
    for i in range(0, len(_signal)):
        if _signal[i] > _threshold:
            if not is_rised:
                rising_edges_timestamps.append(float(_timestamps[i]))
            is_rised = True
        else:
            is_rised = False

    print("Rising edges timestamps")
    print(str(rising_edges_timestamps))

    return rising_edges_timestamps

def find_closest_timestamps_idx(_input_timestamps, _timestamps_list):
    closest_indices = []
    for timestamp in _input_timestamps:
        # Find the position where the timestamp would be inserted to keep the list sorted
        pos = bisect.bisect_left(_timestamps_list, timestamp)
        
        # Check if the timestamp is exactly matched or find the closest one
        if pos == 0:
            closest_indices.append(0)
        elif pos == len(_timestamps_list):
            closest_indices.append(len(_timestamps_list) - 1)
        else:
            # Compare the difference with the previous and next timestamp
            prev_diff = abs(_timestamps_list[pos - 1] - timestamp)
            next_diff = abs(_timestamps_list[pos] - timestamp)
            if prev_diff <= next_diff:
                closest_indices.append(pos - 1)
            else:
                closest_indices.append(pos)
    
    return closest_indices

def get_local_coord(_org, _coord):
    geod = Geod(ellps="WGS84")
    _, _, meters_per_degree_lon = geod.inv(_org[0], _org[1], _org[0] + 1, _org[1])
    _, _, meters_per_degree_lat = geod.inv(_org[0], _org[1], _org[0], _org[1] + 1)

    scale_x = meters_per_degree_lon
    scale_y = meters_per_degree_lat

    scaling_matrix = Affine.scale(scale_x, scale_y)
    translation_matrix = Affine.translation(-_org[0], -_org[1])
    transform = scaling_matrix * translation_matrix

    # print(f"\nTransform:")
    # print(f"|{transform.a}, {transform.b}, {transform.c}|")
    # print(f"|{transform.d}, {transform.e}, {transform.f}|")
    # print(f"|{transform.g}, {transform.h}, {transform.i}|")

    local_x, local_y = transform * (_coord[0], _coord[1])

    return np.array([local_x, local_y, _coord[2]])

def run_ardulog(_filepath_mockup, _filepath_drone):
    type_request = ['RCIN', 'IMU', 'POS', 'BARO', 'MODE', 'MAG', 'XKF1', 'ORGN']
    output = Ardupilot.parse(_filepath_mockup, types=type_request, zero_time_base=True)
    dfs_mockup = output.dfs

    type_request = ['ORGN']
    output = Ardupilot.parse(_filepath_drone, types=type_request, zero_time_base=True)
    dfs_drone = output.dfs

    ### Extract home coordinate ###
    coord_home = np.array([dfs_drone['ORGN']['Lng'], dfs_drone['ORGN']['Lat'], dfs_drone['ORGN']['Alt']])
    coord_home = coord_home.T[-1]

    ### Extract landing positions###
    THRESHOLD = 1200
    landing_timestamps_s = extract_rising_edges(dfs_mockup['RCIN']['timestamp'], dfs_mockup['RCIN']['C5'], THRESHOLD)
    landing_timestamps_f = extract_rising_edges(dfs_mockup['RCIN']['timestamp'], dfs_mockup['RCIN']['C6'], THRESHOLD)

    POS_idx_list_s = find_closest_timestamps_idx(landing_timestamps_s, dfs_mockup['POS']['timestamp'].to_list())
    POS_idx_list_f = find_closest_timestamps_idx(landing_timestamps_f, dfs_mockup['POS']['timestamp'].to_list())

    latitudes_s = dfs_mockup['POS']['Lat'][POS_idx_list_s].to_list()
    longitudes_s = dfs_mockup['POS']['Lng'][POS_idx_list_s].to_list()

    latitudes_f = dfs_mockup['POS']['Lat'][POS_idx_list_f].to_list()
    longitudes_f = dfs_mockup['POS']['Lng'][POS_idx_list_f].to_list()

    XKF1_idx_list_s = find_closest_timestamps_idx(landing_timestamps_s, dfs_mockup['XKF1']['timestamp'].to_list())
    XKF1_idx_list_f = find_closest_timestamps_idx(landing_timestamps_f, dfs_mockup['XKF1']['timestamp'].to_list())

    relalt_s = dfs_mockup['XKF1']['PD'][XKF1_idx_list_s].to_list()
    relalt_f = dfs_mockup['XKF1']['PD'][XKF1_idx_list_f].to_list()

    coords_s = np.array([longitudes_s, latitudes_s, relalt_s]).T
    coords_f = np.array([longitudes_f, latitudes_f, relalt_f]).T

    local_coords_s = []
    for coord_s in coords_s:
        local_coords_s.append(get_local_coord(coord_home, coord_s))

    local_coords_f = []
    for coord_f in coords_f:
        local_coords_f.append(get_local_coord(coord_home, coord_f))

    return local_coords_s, local_coords_f

def add_ardulog(_df):
    df = _df.copy()
    local_coords_s, local_coords_f = run_ardulog(
        Path(__file__).parent / 'inputs/log_1_2025-3-6-15-17-46.bin',
        Path(__file__).parent / 'inputs/log_1_2025-3-6-15-17-46.bin'
    )

    success_values = np.concatenate([np.array([True] * len(local_coords_s)), np.array([False] * len(local_coords_f))])
    combined_local_coords = local_coords_s + local_coords_f

    df = df.loc[df.index.repeat(len(local_coords_s) + len(local_coords_f))].reset_index(drop=True)
    df.insert(0, 'success', success_values)
    df.insert(1, 'landing_x', np.array(combined_local_coords).T[0])
    df.insert(2, 'landing_y', np.array(combined_local_coords).T[1])

    return df

def run_pcl(_args):
    # Path to your compiled executable
    executable_path = "build/pcl"

    # Run the executable with arguments
    try:
        print("Running:", [executable_path] + _args)
        result = subprocess.run(
            [executable_path] + _args,
            check=True,
            text=True,
            capture_output=True
        )
        print("Program output:")
        print(result.stdout)  # Print the standard output of the program
    except subprocess.CalledProcessError as e:
        print("Error running the program:")
        print(e.stderr)  # Print the standard error of the program

def add_pcl(_df):
    pcl_data = []
    for i in range(len(_df)):
        landing_x = str(15.0) + str(i)
        landing_y = str(4.0) + str(i)
        args = [
            "inputs/rtabmap_cloud.ply",
            "outputs/output_pcl.csv",
            landing_x, # str(_df.at[i, 'landing_x'])
            landing_y, # str(_df.at[i, 'landing_y'])
            str(_df.at[i, 'center_x']),
            str(_df.at[i, 'center_y'])
        ]
        run_pcl(args)
        pcl_csv = pd.read_csv("outputs/output_pcl.csv")
        pcl_data.append(pcl_csv.iloc[0].to_dict())

    df_pcl = pd.DataFrame(pcl_data)
    return pd.concat([_df, df_pcl], axis=1)

def main():
    deepforest_csv = {
        'smallest_side': [7.314355765585696],
        'diagonal': [10.344061123713136],
        'area': [53.49980026555672],
        'center_x': [15.50081099],
        'center_y': [3.76794873],
        'specie': ["birch"]
    }

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(deepforest_csv)
    df = add_ardulog(df)
    df = add_pcl(df)

    output_path = "outputs/output.csv"
    df.to_csv(output_path, index=True)



if __name__=="__main__":
    main()
