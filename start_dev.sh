#!/bin/bash
imageId=laion_idle_cah:v0
containerName=laion_cah
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

# define array for docker run call, allows to comment individual arguments
run_args=(
    run
    -it                        # interactive, allocate a pseudo-TTY, detach
    --rm                        # automatically remove the container when it exits
    --net=host                  # use host network
    --name=$containerName       # name of container

    # To restrict GPU availability inside the docker container (e.g. to hide your display GPU) you can use:
    # --gpus '"device=1,2,3"'
    # see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
    --gpus all                  # specify which GPUs to use

    # mount external source file
    --mount type=bind,source="$SCRIPT_DIR/docker/c_h2.py",target="/mnt/spirit/c_h/c_h2.py"

    -w /mnt/spirit/c_h          # set working directory
    --runtime nvidia            # use nvidia runtime
    $imageId
    python3 c_h2.py "$@"        # command to execute
)

echo "Starting docker container attached in dev mode..."
echo "File $SCRIPT_DIR/docker/c_h2.py (outside container) is mapped to /mnt/spirit/c_h/c_h2.py (inside container)"
echo "CTRL-c to kill the script running in the docker container."
echo "CTRL-p CTRL-q (key sequence) to detach (process continues detached in background)."
docker ${run_args[@]}
