FROM ubuntu:22.04 AS base

# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
# Since Ubuntu 24.04, a non-root user is created by default with the name vscode and UID=1000
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
USER $USERNAME

### Install Python essentials
RUN sudo apt-get update && \
    sudo apt-get install -y \
    git \
    python3 \
    python3-pip

### Install C++ essentials
RUN sudo apt-get update && \
    sudo apt-get install -y \
    build-essential \
    gdb

### Install Conda
RUN sudo apt-get update && \
    sudo apt-get install -y \
    wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    sudo /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    sudo sh -c 'echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh' && \
    sudo chown -R $USERNAME:$USERNAME /opt/conda

RUN /opt/conda/bin/conda init bash && \
    echo "conda activate docker" >> ~/.bashrc && \
    /opt/conda/bin/conda create -n docker python=3.10 pandas

ENV PATH=/opt/conda/bin:$PATH
RUN conda create --name docker python=3.10 && \
    conda install pip
