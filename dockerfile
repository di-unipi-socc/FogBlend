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

# Install PyTorch
RUN pip3 install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
RUN pip3 install torch-geometric

# Install PyG extensions
RUN pip3 install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install other dependencies
RUN pip3 install numpy pandas matplotlib seaborn networkx tqdm pyyaml janus_swi 

# Set working directory
WORKDIR /workspace

# Create a user with a specific UID and GID
ARG USER_ID
ARG GROUP_ID

# Create user only if the UID doesn't already exist
RUN if ! getent passwd $USER_ID >/dev/null 2>&1; then \
        if ! getent group $GROUP_ID >/dev/null 2>&1; then \
            groupadd -g $GROUP_ID user; \
        fi && \
        useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash user; \
    else \
        echo "User with UID $USER_ID already exists"; \
    fi

# Switch to user by UID
USER ${USER_ID}

# Default command
CMD ["bash"]