

# Description

This repository contains source code for a model describing boundary current synchronization discussed in [Kohyama et al. (2021)](https://www.science.org/doi/full/10.1126/science.abh3295), which is originally developed by [Gallego and Cessi (2001)](https://doi.org/10.1175/1520-0442(2001)014%3C2815:DVOTOA%3E2.0.CO;2). The implementation is based on [Dr. Kido's repository](https://github.com/shokido/Kohyama2021).

#  Build environment 

## Linux (including WSL)

1. Make `.env`: `$ ./make_env.sh`
2. Install [docker](https://www.docker.com)
3. Build a container image: `$ docker compose build python_linux`
4. Start a container: `$ docker compose up -d python_linux`
5. Connect to JupyterLab [`http://localhost:9999/lab?`](http://localhost:9999/lab?)

## Mac (only for apple silicon)

1. Make `.env`: `$ ./make_env.sh`
2. Install [docker](https://www.docker.com)
3. Build a container image: `$ docker compose build python_mac`
4. Start a container: `$ docker compose up -d python_mac`
5. Connect to JupyterLab [`http://localhost:7777/lab?`](http://localhost:7777/lab?)

# Run the model

1. Start a container and connect to JupyterLab, following the above instructions.
2. Go to `python` directory and start a notebook `run_model.ipynb`
3. Run all the cells in the notebook.