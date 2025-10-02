FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install dependencies for Python
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add the official SWI-Prolog PPA and install the latest version
RUN add-apt-repository ppa:swi-prolog/stable && \
    apt-get update && \
    apt-get install -y swi-prolog && \
    rm -rf /var/lib/apt/lists/*

# Ensure Python points to Python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA
RUN pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip3 install numpy pandas matplotlib seaborn networkx pyyaml tqdm ortools colorama torchopt tensorboard gym==0.22.0 scikit-learn higher swiplserver janus_swi
RUN pip3 install torch-geometric
RUN pip3 install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
RUN pip3 install --force-reinstall scipy

# Create a non-root user with a specific UID and GID
ARG USER_ID
ARG GROUP_ID
RUN groupadd -g $GROUP_ID user && \
    useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash user && \
    usermod -aG sudo user && \
    echo "user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Set working directory and permissions
WORKDIR /workspace
RUN chown -R user:user /workspace

# Switch to non-root user
USER user

# Default command
CMD ["bash"]