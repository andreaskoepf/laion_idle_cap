# NOTE: INTENDED FOR INTERNAL USE ONLY

This is an internal repositroy of [LAION](https://laion.ai/). Using this script outside the LAION cluster will fail.


# Dockerfile for Idle Captioning

This script generates synthetic captions for images of the LAION text-image datasets to utilize GPUs during 'idle' periods.

## Setup & Run

Installing and running the captioning script on a fresh machine:

1. Run `git clone https://github.com/andreaskoepf/laion_idle_cap.git` to clone this repository on the new machine.
2. Run `cd laion_idle_cap` to change to the newly created directory.
3. Run `./install_docker.sh` to install nvidia-docker.
4. Run `./pull.sh` to pull the captioning docker image which contains all dependencies.
5. Run `./start.sh --gpus 0-7 --workers 2` to start the captioning script (detached) in a new docker container. If the `--gpus` option is omitted all available GPUs are used. To select specific devices use comma separated device indices or indice-ranges (e.g. `1-3` or `0,2,4`). The `--workers` option allows to launch more then one worker per GPU (recommended is 2 for full GPU utilization, default is 1).
6. Optionally run `./attach.sh` to attach your terminal to the running instance of the captioning script and see its output.

## Stopping the Docker Container
- run `./stop.sh` or `docker stop laion_cah`

## Other Script Files

- `start_bash.sh` starts the docker container and launches bash (start attached, source will be mounted to `/mnt/src`)
- `start_dev.sh` maps the file `docker/c_h2.py` into a new docker container and starts the script attached (useful for testing changes made outside the docker container, e.g. during development).
- `build.sh` builds the docker image (e.g. laion_idle_cah:v0)
- `save_image.sh` writes the docker image into a tar file
- `push.sh` push the docker container to docker hub
