# Use the NVIDIA CUDA image as the base image
FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

# Install base utilities
RUN apt-get update \
  && apt-get install ffmpeg libsm6 libxext6 -y wget \
  && rm -rf /var/lib/apt/lists/*

# Install miniconda or update if it already exists
ENV CONDA_DIR /opt/conda
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -u -p /opt/conda && \
    rm ~/miniconda.sh

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda
RUN conda init

# Create and activate the conda environment
RUN conda create -n vrd python=3.7

# COPY requirements.txt .
# RUN conda create -n vrd python=3.7 && \
#     /bin/bash -c "source activate vrd && pip install -r requirements.txt"

# Set the default command to run when the container starts
# CMD ["bin/bash"]