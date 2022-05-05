#!/bin/bash

# see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo groupadd docker

# optionally add user to docker group
read -p "Add user '$USER' to 'docker' group? [Y/n]" -n 1 -r yn
echo # newline
case $yn in
    [Nn]* ) ;;
    * ) sudo usermod -aG docker $USER;;
esac

# rastart the docker daemon
sudo systemctl restart docker

# test nvidia-smi in docker container
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
