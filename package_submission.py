import numpy as np

import subprocess

import configparser

# Define the path to your config file
config_file_path = "path/to/your/config.ini"

# Create a ConfigParser object
config_parser = configparser.ConfigParser()

# Read the config file
config_parser.read(config_file_path)

# Extract the desired parameters
team_name = config_parser.get('DEFAULT', 'team-name', fallback='Default Name')
method_name = config_parser.get('DEFAULT', 'method-name', fallback='MyCoolMethod')
prediction_file = config_parser.get('DEFAULT', 'prediction_file', fallback='prediction.npy')

# Command to create a directory named 'submissions'
command = 'mkdir submissions'

# Use subprocess to run the command
try:
    subprocess.run(command, shell=True, check=True)
    print("Directory 'submissions' created successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error creating directory: {e}")

submission_path = "submissions/"+prediction_file
predictions = np.load(submission_path, allow_pickle=True)

submission = {
    "team_name": team_name,
    "method_name": method_name,
    "predictions": predictions,
}

np.save("submissions/submission.npy", submission, allow_pickle=True)