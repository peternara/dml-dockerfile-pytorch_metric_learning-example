FROM nvidia/cuda:9.2-cudnn7-devel

ADD . /work
WORKDIR /work

RUN apt-get update -y
RUN apt-get dist-upgrade -y
RUN apt-get upgrade -y
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
RUN wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz && tar xzf Python-3.7.4.tgz
RUN Python-3.7.4/configure --enable-optimizations
RUN make altinstall
RUN rm -rf Python-3.7.4 && rm Python-3.7.4.tgz
RUN apt-get install -y --no-install-recommends \
    git \
    cmake \
    libblas3 \
    libblas-dev \
    libopencv-dev \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3.7 install pytorch_metric_learning
RUN pip3.7 install --no-cache-dir -r requirements.txt
