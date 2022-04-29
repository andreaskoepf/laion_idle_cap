# Dockerfile for captioning at home

## Script files

- `pull.sh` pull the captioning at home docker container from docker hub 
- `start.sh` starts the docker container and executes the captioning at home python script (`c_h+.py`) (attached/interactive)
- `start_bash.sh` starts the docker container and launches bash (source will be mounted to `/mnt/src`)

- `build.sh` builds the docker image (e.g. laion_idle_cah:v0)
- `save_image.sh` writes the docker image into a tar file
- `push.sh` push the docker container to docker hub
