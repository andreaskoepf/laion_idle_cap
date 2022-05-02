FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

# skip interactive tzdata configuration
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# see https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
# and https://github.com/NVIDIA/nvidia-docker/issues/1631
WORKDIR /tmp
COPY cuda-keyring_1.0-1_all.deb ./
RUN rm /etc/apt/sources.list.d/cuda.list && apt-key del 7fa2af80 && dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && apt-get install rsync git curl wget unzip vim nano less htop build-essential autotools-dev cmake g++ gcc ca-certificates ssh python3-dev libpython3-dev python3-pip -y
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && pip install --upgrade pip

# install PyTorch 1.11.0
RUN python3 -m pip --no-cache-dir install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

WORKDIR /mnt/spirit/c_h

RUN wget --no-check-certificate --no-verbose https://captions.christoph-schuhmann.de/spirit/simplet5-epoch-2-train-loss-0.2212-val-loss-0.2188.zip && unzip simplet5-epoch-2-train-loss-0.2212-val-loss-0.2188.zip && rm simplet5-epoch-2-train-loss-0.2212-val-loss-0.2188.zip

# CLIP
RUN python3 -m pip --no-cache-dir install ftfy regex tqdm git+https://github.com/openai/CLIP.git

# BLIP
RUN git clone -b main https://github.com/LAION-AI/BLIP.git && python3 -m pip --no-cache-dir install -r ./BLIP/requirements.txt

# crawlingathome
RUN git clone -b caption-gen https://github.com/LAION-AI/crawlingathome.git

RUN python3 -m pip --no-cache-dir install webdataset simplet5 awscli pynvml

# copy python scripts
COPY *.py ./

# load models into torch hub cache
RUN python3 prefetch_models.py && rm prefetch_models.py
