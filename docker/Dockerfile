FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

RUN apt-get update \
&& apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

WORKDIR ..

ENV NO_ALBUMENTATIONS_UPDATE=1
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt