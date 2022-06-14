
# Time Series Forecasting

## Project setup

### Prerequisite

1. Python versions and virtual environments are controlled with (the very convenient) [pyenv](https://github.com/pyenv/pyenv). Pyenv installation depends on the OS used:
- MacOS: 
    `brew update`
    `brew install pyenv`
- Windows:
    follow [pyenv-win](https://github.com/pyenv-win/pyenv-win)

2. All the necessary data for this project has been added in git via [Git Large File Storage](https://git-lfs.github.com/). After cloning the project, make sure to first unpack that data: 

    `git lfs install`
    `git lfs pull`  (will download c.a. 720 MB)

### Configure project setup

A virtual environment with all required packages is created by the simple command:

- MacOS: `make setup`
- Windows: `make setup-win` (make sure to use a bash-like shell or follow similar commands)

## Energy Demand Forecasting

Initial discovery of the data is done in `notebooks/`.  

Training is done in stages initiated by a configuration file `src/config.json`. This way of working allows for easy tracking of experiments. The pipeline can be controlled using the corresponding makefile and consist of the following stages:

1. Preprocess

 + Make sure you have setup the right setting in the configuration file.

- MacOS: `make setup`
- Windows: `make setup-win`

2. Train (and validate)

- MacOS: `make train`
- Windows: `make train-win`

 + Track all your experiments with **tensorboard**! In a new shell run: `tensorboard --logdir=logs`

3. Test

The test stage compares different experiments on the test set with a *custom rolling metric*.
The exact test you want to run should be configured inside: `src/test/test_models.py`

- MacOS: `make test`
- Windows: `make test-win`

4. (optional) Animate

- MacOS: `make setup`
- Windows: `make setup-win`

 + *Note: Requires to have ffmpeg installed on your OS (MacOS: `brew install ffmpeg`)*