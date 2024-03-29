#!/bin/bash
imageId=laion_idle_cah:v0
containerName=laion_cah
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

# define array for docker run call, allows to comment individual arguments
run_args=(
    run
    -it                         # interactive, allocate a pseudo-TTY
    --rm                        # automatically remove the container when it exits
    --net=host                  # use host network
    --name=$containerName       # name of container
    -v $SCRIPT_DIR:/mnt/src     # mount source directory

    # To restrict GPU availability inside the docker container (e.g. to hide your display GPU) you can use:
    # --gpus '"device=1,2,3"'
    # see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
    --gpus all                  # specify which GPUs to use

    -w /mnt/spirit/c_h          # set working directory
    --runtime nvidia            # use nvidia runtime
    $imageId
    bash                        # command to execute
)

docker ${run_args[@]}
