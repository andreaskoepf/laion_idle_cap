#!/bin/bash
imageId=laion_idle_cah:v0
containerName=laion_cah

# define array for docker run call, allows to comment individual arguments
run_args=(
    run
    -itd                        # interactive, allocate a pseudo-TTY, detach
    --rm                        # automatically remove the container when it exits
    --net=host                  # use host network
    --name=$containerName       # name of container

    # To restrict GPU availability inside the docker container (e.g. to hide your display GPU) you can use:
    # --gpus '"device=1,2,3"'
    # see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
    --gpus all                  # specify which GPUs to use

    -w /mnt/spirit/c_h          # set working directory
    --runtime nvidia            # use nvidia runtime
    $imageId
    python3 c_h2.py             # command to execute
)

echo "Starting docker container detached in background mode..."
echo "Use 'docker attach $containerName' or './attach.sh' to attach to the process output."
echo "Container ID:"
docker ${run_args[@]}
