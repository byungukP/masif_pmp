FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Dockerfile to build the environment for MaSIF-PMP
# Python 3.10.13, PyTorch v2.1.2, CUDA 12.1.1, cuDNN 8.9.0, Ubuntu 20.04, linux/amd64
# ByungUk Park, UW-Madison, 2025

ENV TZ="America/Chicago"
WORKDIR /root/
ARG BRANCH="main"
ARG NUM_CORES=2
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y build-essential \
                   gcc-9 \
                   g++-9 \
                   libgmp-dev \
                   libmpfr-dev \
                   libgmpxx4ldbl \
                   libboost-dev \
                   libboost-thread-dev \
                   zip unzip patchelf \
                   wget git vim nano cmake libgl1-mesa-glx dssp curl python2.7 && \
    apt clean
### need python2.7 for pdb2pqr installation

# DOWNLOAD/INSTALL PYMESH (v0.3.1)
RUN git clone --single-branch -b $BRANCH https://github.com/nuvolos-cloud/PyMesh.git

ENV PYMESH_PATH /root/PyMesh
ENV NUM_CORES $NUM_CORES
WORKDIR $PYMESH_PATH

RUN git submodule update --init && \
    pip install -r $PYMESH_PATH/python/requirements.txt && \
    ./setup.py bdist_wheel && \
    rm -rf build_3.7 third_party/build && \
    python $PYMESH_PATH/docker/patches/patch_wheel.py dist/pymesh2*.whl && \
    pip install --upgrade pip && \
    pip install dist/pymesh2*.whl && \
    python -c "import pymesh; pymesh.test()"

# DOWNLOAD/INSTALL APBS (v1.5), MSMS (v2.6.1)
RUN mkdir /install
WORKDIR /install
RUN git clone https://github.com/Electrostatics/apbs-pdb2pqr
WORKDIR /install/apbs-pdb2pqr
RUN git checkout b3bfeec && \
    git submodule init && \
    git submodule update && \
    cmake -DGET_MSMS=ON apbs && \
    make && \
    make install && \
    cp -r /install/apbs-pdb2pqr/apbs/externals/mesh_routines/msms/msms_i86_64Linux2_2.6.1 /root/msms/ && \
    curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py && \
    python3 get-pip.py

# INSTALL PDB2PQR
WORKDIR /install/apbs-pdb2pqr/pdb2pqr
RUN git checkout b3bfeec && \
    python2.7 scons/scons.py install

# Setup environment variables 
ENV MSMS_BIN /usr/local/bin/msms
ENV APBS_BIN /usr/local/bin/apbs
ENV MULTIVALUE_BIN /usr/local/share/apbs/tools/bin/multivalue
ENV PDB2PQR_BIN /root/pdb2pqr/pdb2pqr.py

# DOWNLOAD reduce (for protonation) (v4.14)
ARG REDUCE_TAG=v4.14
WORKDIR /install
RUN git clone https://github.com/rlabduke/reduce.git
WORKDIR /install/reduce
RUN git fetch --all --tags && \
    git checkout tags/${REDUCE_TAG} -b build-${REDUCE_TAG} && \
    make install && \
    mkdir -p /install/reduce/build/reduce
WORKDIR /install/reduce/build/reduce
RUN cmake /install/reduce/reduce_src
WORKDIR /install/reduce/reduce_src
RUN make && \
    make install

# Install python libraries
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Clone masif
WORKDIR /
RUN git clone --single-branch -b $BRANCH https://github.com/byungukP/masif_pmp.git
WORKDIR /masif_pmp

CMD [ "bash" ]
