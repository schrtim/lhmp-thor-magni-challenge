import numpy as np

import subprocess

# Command to create a directory named 'submissions'
command = 'mkdir submissions'

# Use subprocess to run the command
try:
    subprocess.run(command, shell=True, check=True)
    print("Directory 'submissions' created successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error creating directory: {e}")

predictions = np.load("submissions/test_prediction_cvm_nr1.npy", allow_pickle=True)

submission = {
    "team_name": "team_name",
    "method_name": "method_name",
    "predictions": predictions,
}

np.save("submissions/submission.npy", submission, allow_pickle=True)