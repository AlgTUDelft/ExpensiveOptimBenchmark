FROM frehbach/cfd-test-problem-suite

# Install python 3.7
RUN apt-get install -y software-properties-common &&\
    apt-get update &&\
    apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev &&\
    wget https://www.python.org/ftp/python/3.7.10/Python-3.7.10.tgz &&\
    tar -xf Python-3.7.10.tgz &&\
    cd Python-3.7.10 &&\
    # With optimizations takes longer to build, but given the time senstive nature of the benchmarks
    # it is likely good to enable them.
    ./configure --enable-optimizations --with-ensurepip=install &&\
    # Alternatively, for testing if things work at all:
    # ./configure --with-ensurepip=install &&\
    make -j 4 &&\
    make altinstall &&\
    cd .. &&\
    # apt-get install -y python3-pip python3.7 python3.7-dev swig &&\
    python3.7 -m ensurepip --default-pip &&\
    apt-get install -y swig &&\
    python3.7 -m pip install --upgrade pip &&\
    python3.7 -m pip install --upgrade cython setuptools wheel

# Install benchmark dependencies.
COPY ./requirements.txt ./requirements_eob.txt

RUN python3.7 -m pip install -r requirements_eob.txt

COPY expensiveoptimbenchmark ./expensiveoptimbenchmark

RUN bash ./expensiveoptimbenchmark/problems/ESP2/patch.sh

RUN ln -s ./dockerCall.sh ./evaluate.sh

# ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
