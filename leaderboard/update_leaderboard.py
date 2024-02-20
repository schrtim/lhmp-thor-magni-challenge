import json

def update_leaderboard(new_entry):
    try:
        # Load existing leaderboard data from JSON file
        with open('leaderboard/leaderboard.json', 'r') as f:
            leaderboard_data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, initialize leaderboard data as an empty list
        leaderboard_data = []

    # Append the new entry to the leaderboard data
    leaderboard_data.append(new_entry)

    # Sort the leaderboard data by score (assuming each entry has a 'score' key)
    leaderboard_data.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Write the updated leaderboard data back to the JSON file
    with open('leaderboard/leaderboard.json', 'w') as f:
        json.dump(leaderboard_data, f, indent=4)

if __name__ == "__main__":
    with open("leaderboard/board_entry.json", "r") as f:
        new_entry = json.load(f)

        # Update the leaderboard with the new entry
        update_leaderboard(new_entry)
