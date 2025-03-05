FROM python:3.10

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

ARG WS=/home/$USERNAME/python_docker/

#DeepForest
WORKDIR $WS

RUN sudo apt-get update -y && \
    sudo apt-get install -y \
    libgl1

RUN git clone https://github.com/weecology/DeepForest.git

WORKDIR $WS/DeepForest/
RUN pip install --upgrade pip && \
    pip install -r dev_requirements.txt && \
    pip install -e . && \
    pip install imagecodecs && \
    pip install tifffile && \
    pip install rasterio
