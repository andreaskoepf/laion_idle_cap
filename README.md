# Dockerfile for captioning at home

## Setup & run

Running the captioning script on a fresh machine:

1. `./install_docker.sh` - install nvidia-docker 
2. `./pull.sh` - pull docker image
3. `./start.sh` - start captioning script (detached) in docker container
4. `./attach.sh` attach terminal to a running instance of the captioning script


## Other script files

- `start_bash.sh` starts the docker container and launches bash (start attached, source will be mounted to `/mnt/src`)
- `build.sh` builds the docker image (e.g. laion_idle_cah:v0)
- `save_image.sh` writes the docker image into a tar file
- `push.sh` push the docker container to docker hub
