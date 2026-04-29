FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip3 install psutil

WORKDIR /app

# Copy the entire workspace into the container
COPY . /app/

# Build the HGS-CVRP algorithm
WORKDIR /app/baselines/HGS-CVRP
RUN mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" && make bin

# Reset working directory to the app root
WORKDIR /app

# The default command runs the benchmark script
CMD ["python3", "scripts/run_benchmarks_hgs.py", "--folder", "data/instances", "--time-limit", "2700"]
