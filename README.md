# THÖR-MAGNI Challenge 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10407222.svg)](https://zenodo.org/doi/10.5281/zenodo.10407222)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://magni-dash.streamlit.app)

Results will be displayed in the [Leaderboard](https://schrtim.github.io/lhmp-thor-magni-challenge/leaderboard/leaderboard.html)

<img src="assets/Logo.svg" align="left" width=25% height=25%>

Explore the wealth of insights within the THÖR-MAGNI dataset, a large-scale collection of human and robot navigation and interaction data. This dataset extends THÖR with 3.5 times more motion data, introduces novel interactive scenarios, and includes rich contextual annotations. THÖR-MAGNI offers diverse navigation styles of both mobile robots and humans engaged in shared environments with robotic agents, featuring multi-modal data for a comprehensive representation. Accessible online at [thor.oru.se](http://thor.oru.se/), THÖR-MAGNI serves as a valuable resource for training activity-conditioned motion prediction models and investigating visual attention during human-robot interaction.

To further support researchers, THÖR-MAGNI comes with a dedicated set of user-friendly tools, including a [dashboard](https://magni-dash.streamlit.app) and the specialized Python package [thor-magni-tools](https://github.com/tmralmeida/thor-magni-tools). These tools streamline the visualization, filtering, and preprocessing of raw trajectory data, enhancing the accessibility and usability of the dataset. By providing these resources, we aim to equip researchers with versatile and efficient tools to navigate, analyze, and extract valuable insights from the dataset.

Already employed in research papers, the THÖR-MAGNI dataset demonstrates its usefulness for advancing the field of human-robot interaction, offering a wealth of opportunities for predictive modeling and comprehensive exploration of human motion dynamics in various scenarios.

Results will be displayed in the [Leaderboard](https://schrtim.github.io/lhmp-thor-magni-challenge/leaderboard/leaderboard.html)

## How to get the data?

Our dataset is uploaded and publicly available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10407222)

To download it and get hands on with the data, you can simply curl it:
```
curl -O https://zenodo.org/records/10407223/files/THOR_MAGNI.zip && unzip -d . THOR_MAGNI.zip && rm -rf THOR_MAGNI.zip
```

## Submission Format

You train and develop your method locally and generate prediction files, that can be packaged and submitted to our challenge.

### Information for the user

Submissions to our challenge are only to be made in numpys [**.npy** format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).

We provide a sample jupyter notebook called **run_simple_cvm** that demonstrates the format for predictions.

The structure of the predictions file is as follows:

[
- `predictions_scenario_1` (numpy array): Array of predicted trajectories for the first scenario.
  - `trajectory_1` (numpy array): Predicted trajectory for the first scenario.
    - `time_step_1` (numpy array): Coordinates (x, y) at time step 1.
    - `time_step_2` (numpy array): Coordinates (x, y) at time step 2.
    - ...
    - `time_step_N` (numpy array): Coordinates (x, y) at the final time step.
  - `trajectory_N` (numpy array): Predicted trajectory for the first scenario.
    - `time_step_1` (numpy array): Coordinates (x, y) at time step 1.
    - `time_step_2` (numpy array): Coordinates (x, y) at time step 2.
    - ...
    - `time_step_N` (numpy array): Coordinates (x, y) at the final time step.
- `predictions_scenario_N` (numpy array): Array of predicted trajectories for the Nth scenario.
  - ...

]

### Adjust submission metadata

The repositories main directory contains a **config.yml** file.
Here you can adjust your team name and specifiy your method.
As well as specify the name of your prediction file that you want to upload to the challenge as a prediction in the next step.

## How to make a submission

To participate in this challenge, follow these steps:

1. Fork this repository to your own GitHub account.
2. Clone the forked repository to your local machine.
3. Create a conda environment using the following command (replace env_name with the name you want)
```
conda create --name <env_name> python=3.10.8 && conda activate <env_name> && pip install -r requirements.txt
```

3. Copy your submission_file.npy in the submissions folder and package it. 
(NOTE: This will use the metadata you specified in config.yml)

```
python package_submission.py
```

4. Commit and push ONLY the *submission.npy* file to your forked repository.
5. Create a pull request to submit your *submissions.npy* file to the **challenge branch**. Your pull request will be inspected by one of our admins and approved if there are no outstanding issues.

## Terms and Conditions

Note that the ground truth test annotations are provided in the data loader. This is because they match the ground truth of the original THÖR-MAGNI data, which is readily available. We trust participants to not utilize these unethically, especially as we will be inviting the top participants to present their work at our 2024 ICRA workshop and will review submissions accordingly. For participation in the workshop, only submissions provided before TBA will be considered. Top performers will then be contacted to validate their approaches and provide instructions for submitting their writeup for the TBA workshop.

Namely, we expect these train/val/test paradigms to be followed:

## Contact

If you have questions or remarks regarding this challenge, please contact one of our team members:
- [Tim Schreiter](github.com/schrtim)
- [Janik Kaden](github.com/janikkaden)
- [Tiago Almeida](github.com/tmralmeida)