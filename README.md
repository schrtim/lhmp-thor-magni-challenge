# THÖR-MAGNI Challenge 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10407222.svg)](https://zenodo.org/doi/10.5281/zenodo.10407222)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://magni-dash.streamlit.app)

<span style="font-size:3em;">Results will be displayed on the [Leaderboard](https://schrtim.github.io/lhmp-thor-magni-challenge/leaderboard/leaderboard.html)</span>


<img src="assets/Logo.svg" align="right" width=25% height=25%> 

## About this repository

This repository is for you if you want to partake in the THÖR-MAGNI challenge.
Develop, train and test your own methods with the dataset.

For all this we provide you with a comprehensive [individual repository](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras), that contains a sample dataloader. Furthermore, the repository describes everything you need to know about the handling of the THÖR-MAGNI data. For a first impression of how the data looks like, you can use our [visualization tool](https://magni-dash.streamlit.app)<br />

## 1. Checkout the Benchmark repo for dataloaders, sample models and a predefined train/test split

[**BENCHMARK REPO**](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras)

## 2. Submission Format

You train and develop your method locally and generate prediction files, that can be packaged and submitted to our challenge.

### 2.1. Information for the user

Submissions to our challenge are only to be made in [**.npy** format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).

For information on how to format your predictions, before proceeding with th next steps, please checkout the [**BENCHMARK REPO**](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras) once again.

### 2.2. Adjust submission metadata

The repository's main directory contains a **config.ini** file.
Here you can adjust your team name and specify your method.
Also, specify the name of your prediction file that you want to upload to the challenge as a prediction in the next step.

## 3. How to test a prediction

To participate in this challenge, follow these steps:

1. Fork this repository to your own GitHub account.
2. Clone the forked repository to your local machine.
3. Create a conda environment using the following command:
```
conda env create -f environment.yaml && conda activate thor-magni-challenge
```

4. Copy your submission file (.npy) into the repo base folder and package it:
(NOTE: This will use the metadata you specified in config.yml and create a submission.npy file inside the submissions folder.)

```
python package_submission.py
```

5. To test your challenge results, you can run the processing script locally. This will print the leaderboard entry for the previously packaged submission.
```
python challenge_processing_script.py
```
## 4. Make a submission to our challenge 

<span style="font-size:1.5em;">**Please proceed only with these steps if you want to submit your final results!**</span>


6. Commit and push **ONLY** the *submission.npy* file inside the submissions folder to your forked repository.
7. Create a pull request to submit your *submissions.npy* file to the **challenge branch**. Your pull request will be inspected by one of our admins and approved if there are no outstanding issues.

## 5. Terms and Conditions

Note that the ground truth test annotations are provided in the [**BENCHMARK REPO**](https://github.com/tmralmeida/lhmp-thor-magni-challenge-extras). This is because they match the ground truth of the original THÖR-MAGNI data, which is readily available. We trust participants not to utilize these unethically, especially as we will be inviting the top participants to present their work at our 2024 ICRA workshop and will review submissions accordingly. For participation in the workshop, only submissions provided before **01.05.** will be considered. Top performers will then be contacted to validate their approaches and provide instructions for submitting their writeup for the **6th Workshop on Longterm Human Motion Prediction (LHMP)** at the **13.05.2024** workshop.

## Contact

If you have questions or remarks regarding this challenge, please contact one of our team members:
- [Tim Schreiter](http://github.com/schrtim)
- [Janik Kaden](http://github.com/janikkaden)
- [Tiago Almeida](http://github.com/tmralmeida)

## About the dataset

The THÖR-MAGNI dataset is a large-scale collection of human and robot navigation and interaction data. THÖR-MAGNI offers diverse navigation styles of both mobile robots and humans engaged in shared environments with robotic agents, featuring multi-modal data for a comprehensive representation. THÖR-MAGNI serves as a valuable resource for training activity-conditioned motion prediction models and investigating visual attention during human-robot interaction.

To further support researchers, THÖR-MAGNI comes with a dedicated set of user-friendly tools, including a [dashboard](https://magni-dash.streamlit.app) and the specialized Python package [thor-magni-tools](https://github.com/tmralmeida/thor-magni-tools). These tools streamline the visualization, filtering, and preprocessing of raw trajectory data, enhancing the accessibility and usability of the dataset. By providing these resources, we aim to equip researchers with versatile and efficient tools to navigate, analyze, and extract valuable insights from the dataset.
