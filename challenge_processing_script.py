import importlib
import os
import sys
import json

import numpy as np

import evaluator


def get_curr_working_dir():
    curr_working_dir = os.getcwd()
    return curr_working_dir


def run():
    current_working_directory = get_curr_working_dir()
    sys.path.append("{}".format(current_working_directory))

    annotation_file_path = "{}/annotations/test_annotations.npy".format(
        current_working_directory
    )  # Add the test annotation file path
    user_submission_file_path = "{}/submissions/submission.npy".format(
        current_working_directory
    )  # Add the sample submission file path

    submitted = np.load(user_submission_file_path, allow_pickle=True)
    team_name = submitted.item().get("team_name")
    method_name = submitted.item().get("method_name")
    predictions = submitted.item().get("predictions")


    print("Trying to evaluate")

    result = evaluator.evaluate(annotation_file_path,predictions)
    
    leaderboard_entry = {"team_name": team_name, "method_name": method_name, "result": result}

    print("Leaderboard Entry: ", leaderboard_entry)
    
    print("Evaluated Successfully!")

    print("Creating leaderboard entry")

    with open("leaderboard/board_entry.json", "w") as f:
        json.dump(leaderboard_entry, f)


if __name__ == "__main__":
    run()
