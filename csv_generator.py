import subprocess

import pandas as pd

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


def main():
    # Create a dictionary with the given values
    data = {
        'smallest_side': [7.314355765585696],
        'diagonal': [10.344061123713136],
        'area': [53.49980026555672],
        'center_x': [15.50081099],
        'center_y': [3.76794873],
        'landing_x': [15.0],
        'landing_y': [4.0],
        'specie': ["birch"]
    }

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame
    print("Original DataFrame:")
    print(df)

    args = [
        "inputs/rtabmap_cloud.ply",  # PLY file path
        str(df.at[0, 'landing_x']),
        str(df.at[0, 'landing_y']),
        str(df.at[0, 'center_x']),
        str(df.at[0, 'center_y']),
        "outputs/output_pcl.csv"         # Output CSV path
    ]
    run_pcl(args)

    # Load the CSV file into a DataFrame
    df_pcl = pd.read_csv("outputs/output_pcl.csv")

    # Display the DataFrame
    print("\nConcatenated DataFrame:")
    print(df_pcl)

        # Concatenate the two DataFrames along columns
    concatenated_df = pd.concat([df, df_pcl], axis=1)

    # Display the concatenated DataFrame
    print("\nConcatenated DataFrame:")
    print(concatenated_df)

    # Save the concatenated DataFrame to a new CSV file
    output_path = "outputs/concatenated_output.csv"
    concatenated_df.to_csv(output_path, index=False)
    print(f"\nConcatenated DataFrame saved to {output_path}")



if __name__=="__main__":
    main()
