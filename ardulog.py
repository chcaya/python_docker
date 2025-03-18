# https://github.com/PyFlightCoach/ArdupilotLogReader

from ardupilot_log_reader.reader import Ardupilot
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import bisect

import open3d as o3d
import laspy


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


### Extract and print logs ###
type_request = ['RCIN', 'IMU', 'POS', 'BARO', 'MODE', 'MAG', 'XKF1', 'ORGN']
output = Ardupilot.parse(Path(__file__).parent / 'inputs/log_1_2025-3-6-15-17-46.bin', types=type_request, zero_time_base=True)
print(str(output.dfs['RCIN']['C5']))
print(str(output.dfs['RCIN']['C6']))
print(str(output.dfs['IMU']))
print(str(output.dfs['POS']))
print(str(output.dfs['MAG']))
print(str(output.dfs['XKF1']))
print(str(output.dfs['ORGN']))


### Extract landing positions###
c5 = output.dfs['RCIN']['C5']
c6 = output.dfs['RCIN']['C6']
rcin_timestamps = output.dfs['RCIN']['timestamp']
THRESHOLD = 1200

landing_timestamps_s = extract_rising_edges(rcin_timestamps, c5, THRESHOLD)
landing_timestamps_f = extract_rising_edges(rcin_timestamps, c6, THRESHOLD)

POS_timestamps = output.dfs['POS']['timestamp'].to_list()
POS_idx_list_s = find_closest_timestamps_idx(landing_timestamps_s, POS_timestamps)
POS_idx_list_f = find_closest_timestamps_idx(landing_timestamps_f, POS_timestamps)

latitudes_s = output.dfs['POS']['Lat'][POS_idx_list_s].to_list()
longitudes_s = output.dfs['POS']['Lng'][POS_idx_list_s].to_list()
altitudes_s = output.dfs['POS']['RelHomeAlt'][POS_idx_list_s].to_list()

latitudes_f = output.dfs['POS']['Lat'][POS_idx_list_f].to_list()
longitudes_f = output.dfs['POS']['Lng'][POS_idx_list_f].to_list()
altitudes_f = output.dfs['POS']['RelHomeAlt'][POS_idx_list_f].to_list()


XKF1_timestamps = output.dfs['XKF1']['timestamp'].to_list()
XKF1_idx_list_s = find_closest_timestamps_idx(landing_timestamps_s, XKF1_timestamps)
XKF1_idx_list_f = find_closest_timestamps_idx(landing_timestamps_f, XKF1_timestamps)

x_s = output.dfs['XKF1']['PE'][XKF1_idx_list_s].to_list()
y_s = output.dfs['XKF1']['PN'][XKF1_idx_list_s].to_list()
relalt_s = output.dfs['XKF1']['PD'][XKF1_idx_list_s].to_list()

x_f = output.dfs['XKF1']['PE'][XKF1_idx_list_f].to_list()
y_f = output.dfs['XKF1']['PN'][XKF1_idx_list_f].to_list()
relalt_f = output.dfs['XKF1']['PD'][XKF1_idx_list_f].to_list()


### Figures ###
plt.figure()
plt.plot(output.dfs['RCIN']['timestamp'].to_numpy(), output.dfs['RCIN']['C5'].to_numpy(), color='r', label='C5')
plt.plot(output.dfs['RCIN']['timestamp'].to_numpy(), output.dfs['RCIN']['C6'].to_numpy(), color='g', label='C6')
plt.xlabel("Time (sec)")
plt.ylabel("RCIN")
plt.legend()
plt.title('Button inputs')

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(output.dfs['POS']['timestamp'].to_numpy(), output.dfs['POS']['Lat'].to_numpy(), color='r', label='Lat')
plt.plot(output.dfs['POS']['timestamp'].to_numpy(), output.dfs['POS']['Lng'].to_numpy(), color='g', label='Lng')
plt.xlabel("Time (sec)")
plt.ylabel("POS")
plt.legend()
plt.title('Lattitude and Longitude')

plt.subplot(1, 2, 2)
plt.plot(output.dfs['POS']['timestamp'].to_numpy(), output.dfs['POS']['Alt'].to_numpy(), color='b', label='Alt')
plt.xlabel("Time (sec)")
plt.ylabel("Altitude (cm)")
plt.legend()
plt.title('POS Altitude')

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(output.dfs['XKF1']['timestamp'].to_numpy(), output.dfs['XKF1']['PN'].to_numpy(), color='r', label='North')
plt.plot(output.dfs['XKF1']['timestamp'].to_numpy(), output.dfs['XKF1']['PE'].to_numpy(), color='g', label='East')
plt.xlabel("Time (sec)")
plt.ylabel("XKF1")
plt.legend()
plt.title('North and East')

plt.subplot(1, 2, 2)
plt.plot(output.dfs['XKF1']['timestamp'].to_numpy(), output.dfs['XKF1']['PD'].to_numpy(), color='b', label='Down')
plt.xlabel("Time (sec)")
plt.ylabel("Altitude (cm)")
plt.legend()
plt.title('XKF1 Altitude')

plt.figure()
plt.plot(latitudes_s, longitudes_s, marker='o', linestyle='none', color='g', label='Success')
plt.plot(latitudes_f, longitudes_f, marker='o', linestyle='none', color='r', label='Fail')
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.legend()
plt.title('Landings')

# plt.show()

# # Combine latitude, longitude, and altitude into 3D points
# points_s = np.column_stack((x_s, y_s, relalt_s))  # Green points
# points_f = np.column_stack((x_f, y_f, relalt_f))  # Red points

# # Create an Open3D point cloud
# point_cloud = o3d.geometry.PointCloud()

# # Add points to the point cloud
# point_cloud.points = o3d.utility.Vector3dVector(np.vstack((points_s, points_f)))

# # Assign colors: green for points_s, red for points_f
# colors = np.vstack((
#     np.tile([0, 1, 0], (len(points_s), 1)),  # Green for points_s
#     np.tile([1, 0, 0], (len(points_f), 1))   # Red for points_f
# ))
# point_cloud.colors = o3d.utility.Vector3dVector(colors)

# # Visualize the point cloud
# # o3d.visualization.draw_geometries([point_cloud])

# output_file = "landings_cloud.ply"
# o3d.io.write_point_cloud(output_file, point_cloud)


# Combine the data into a single array
latitudes = np.array(latitudes_s + latitudes_f)
longitudes = np.array(longitudes_s + longitudes_f)
altitudes = np.array(altitudes_s + altitudes_f)

# Create RGB colors (green for 's', red for 'f')
red = np.array([255, 0, 0])  # Red
green = np.array([0, 255, 0])  # Green

# Create an array for colors
colors = np.vstack([
    np.tile(green, (len(latitudes_s), 1)),  # Green for 's'
    np.tile(red, (len(latitudes_f), 1))     # Red for 'f'
])

# Create a new LAS file
header = laspy.LasHeader(version="1.4", point_format=7)  # Point format 7 supports RGB
header.scale = (0.0000001, 0.0000001, 0.0000001)  # Scale for x, y, z
las = laspy.LasData(header)

# Assign coordinates and colors
las.x = longitudes
las.y = latitudes
las.z = altitudes

las.red = colors[:, 0]
las.green = colors[:, 1]
las.blue = colors[:, 2]

# Save the LAS file
las.write("outputs/landings.las")

print("LAS file created successfully!")
