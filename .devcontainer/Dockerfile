FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

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

# Install Python essentials
RUN sudo apt-get update && \
    sudo apt-get install -y \
    git \
    python3 \
    python3-pip

ARG WS=/home/$USERNAME/python_docker/

WORKDIR $WS

RUN pip install GDAL=3.10.2 && \
    pip install torch torchvision torchaudio && \
    pip install git+https://github.com/PatBall1/detectree2.git
