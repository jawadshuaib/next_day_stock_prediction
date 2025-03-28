"""
--
-- Consolidate all JSON files into a single pickle file --
--
This script processes JSON files containing stock data, cleans the data, and saves it as a consolidated pickle file.
Workflow:
    1. Retrieves all JSON files from the specified directory (`data/daily/existing/*.json`).
    2. Iterates through each JSON file:
        - Extracts the ticker name from the filename.
        - Loads and parses the JSON data.
        - Converts the data into a pandas DataFrame.
        - Adds a "ticker" column and converts the "date" column to datetime format.
        - Drops rows with invalid or missing dates.
    3. Combines all individual DataFrames into a single DataFrame.
    4. Cleans the combined DataFrame:
        - Sorts by ticker and date.
        - Handles invalid values (e.g., NaN, infinity).
        - Ensures numeric columns are properly formatted.
        - Drops rows with any remaining NaN values.
    5. Saves the cleaned DataFrame as a pickle file (`data/daily_stocks.pkl`).
Error Handling:
    - Catches and logs errors related to missing keys, invalid values, or JSON decoding issues.
Output:
    - A cleaned and consolidated pickle file containing stock data.
Usage:
    - Ensure the input JSON files are located in the `data/daily/existing/` directory.
    - Run the script to process the files and generate the output pickle file.
"""
import glob
import json
import os
import numpy as np
import pandas as pd

# Output file path
output_file = "data/daily_stocks.pkl"

# Get all JSON files in the specified directory
json_files = glob.glob("data/daily/existing/*.json")

# Initialize an empty list to store individual dataframes
dfs = []

# Process each JSON file
for file_path in json_files:
    try:
        # Extract ticker name from the filename
        ticker = os.path.basename(file_path).split('.')[0]
        
        # Load JSON data
        with open(file_path, 'r') as file:
            json_string = file.read()
        data = json.loads(json_string)
        
        # Create a dataframe from the JSON data
        temp_df = pd.DataFrame(data["data"])
        
        # Add a ticker column
        temp_df["ticker"] = ticker
        
        # Convert the "date" column to datetime format
        temp_df["date"] = pd.to_datetime(temp_df["date"], errors='coerce')
        
        # Drop rows with invalid or missing dates
        temp_df.dropna(subset=["date"], inplace=True)
        
        # Append the dataframe to the list
        dfs.append(temp_df)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"Error processing file {file_path}: {e}")

# Combine all dataframes into a single dataframe
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort the dataframe by ticker and date
    combined_df = combined_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Store the ticker as both a column and an index
    combined_df["ticker_name"] = combined_df["ticker"]  # Keep a copy in a column
    combined_df.set_index("ticker", inplace=True)
    
    # Handle invalid values
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
    combined_df.dropna(inplace=True)  # Drop rows with NaN values
    
    # Ensure all numeric columns are properly formatted
    for col in combined_df.select_dtypes(include=[np.number]).columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    # Drop rows with any remaining NaN values
    combined_df.dropna(inplace=True)
    
    # Save the cleaned dataframe as a pickle file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_pickle(output_file)
    
    print(f"Processed {len(json_files)} ticker files and saved to {output_file}")
else:
    print("No JSON files found in the specified directory.")