FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu20.04 
ARG DEBIAN_FRONTEND=noninteractive

# Set the workdir in the docker container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libsqlite3-dev \
    lzma \
    liblzma-dev \
    libbz2-dev \
    git \
    wget

# Download and install Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations --enable-loadable-sqlite-extensions && \
    make && make install

# Install pip for Python 3.10
RUN python3.10 -m ensurepip --upgrade && \
    python3.10 -m pip install --no-cache-dir --upgrade pip

# Copy requirements.txt from your project folder and install Python dependencies
COPY requirements.txt .
RUN pip3.10 install --no-cache-dir -r requirements.txt
RUN pip3.10 install db-sqlite3

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port uWSGI will listen on
EXPOSE 8000

# Run the application:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]