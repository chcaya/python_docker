from ardupilot_log_reader.reader import Ardupilot
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

type_request = ['RCIN', 'IMU', 'GPS', 'BARO', 'MODE', 'MAG']
output = Ardupilot.parse(Path(__file__).parent / 'log_1_2025-3-6-15-17-46.bin', types=type_request, zero_time_base=True)
print(str(output.dfs['RCIN']['C5']))
print(str(output.dfs['RCIN']['C6']))
print(str(output.dfs['IMU']))
print(str(output.dfs['GPS']))
print(str(output.dfs['MAG']))

plt.figure()
plt.plot(output.dfs['RCIN']['timestamp'].to_numpy(), output.dfs['RCIN']['C5'].to_numpy(), color='r', label='C5')
plt.plot(output.dfs['RCIN']['timestamp'].to_numpy(), output.dfs['RCIN']['C6'].to_numpy(), color='g', label='C6')
plt.xlabel("Time (sec)")
plt.ylabel("RCIN")
plt.legend()
plt.title('Button inputs')

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(output.dfs['GPS']['timestamp'].to_numpy(), output.dfs['GPS']['Lat'].to_numpy(), color='r', label='Lat')
plt.plot(output.dfs['GPS']['timestamp'].to_numpy(), output.dfs['GPS']['Lng'].to_numpy(), color='g', label='Lng')
plt.xlabel("Time (sec)")
plt.ylabel("GPS")
plt.legend()
plt.title('Lattitude and Longitude')

plt.subplot(1, 2, 2)
plt.plot(output.dfs['GPS']['timestamp'].to_numpy(), output.dfs['GPS']['Alt'].to_numpy(), color='b', label='Alt')
plt.xlabel("Time (sec)")
plt.ylabel("Altitude (cm)")
plt.legend()
plt.title('GPS Altitude')

plt.show()
