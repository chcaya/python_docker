import subprocess
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Path to your compiled executable
executable_path = "build/pcl"

# Arguments to pass to the executable
args = [
    "inputs/rtabmap_cloud.ply",  # PLY file path
    "15.0",                      # Landing point X
    "4.0",                       # Landing point Y
    "15.50081099",               # Center point X
    "3.76794873",                # Center point Y
    "outputs/output.csv"         # Output CSV path
]

# Run the executable with arguments
try:
    print("Running:", [executable_path] + args)
    result = subprocess.run(
        [executable_path] + args,
        check=True,
        text=True,
        capture_output=True
    )
    print("Program output:")
    print(result.stdout)  # Print the standard output of the program
except subprocess.CalledProcessError as e:
    print("Error running the program:")
    print(e.stderr)  # Print the standard error of the program
