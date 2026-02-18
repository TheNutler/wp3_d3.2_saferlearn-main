FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Fix for nvidia images
# RUN apt-key del 7fa2af80
# RUN apt-key del 3bf863cc
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# install binaries dependencies
RUN apt-get -yqq update && apt-get install -yqq --no-install-recommends\
        g++ \
        automake \
        build-essential \
        ca-certificates \
        git \
        libboost-dev \
        libboost-thread-dev \
        libntl-dev \
        libsodium-dev \
        libssl-dev \
        libtool \
        m4 \
        python3 \
        python3-dev \
        python3-pip \
        texinfo \
        yasm \
        curl \
        openjdk-8-jdk \
        pkg-config \
        swig \
        unzip \
        wget \
        zlib1g-dev \
        zip

RUN pip3 install --upgrade pip setuptools

# Install Python dependencies for ML
COPY requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

ENV MPC_DIR=/MPC

# Copy the directory
COPY MP-SPDZ-source ${MPC_DIR}

# Compile the library
WORKDIR ${MPC_DIR}
RUN make -j tldr

# Compile the teachers executables
COPY pate-teacher.cpp ${MPC_DIR}/ExternalIO
COPY pate-teacher-one-hot.cpp ${MPC_DIR}/ExternalIO
COPY privacy-guardian.cpp ${MPC_DIR}/ExternalIO
RUN make -j externalIO

COPY pate_aggregation.mpc ${MPC_DIR}/Programs/Source
COPY pate_aggregation_privacy_guardian.mpc ${MPC_DIR}/Programs/Source
COPY pate_aggregation_one_hot.mpc ${MPC_DIR}/Programs/Source

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
