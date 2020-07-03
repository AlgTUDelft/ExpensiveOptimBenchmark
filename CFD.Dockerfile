FROM frehbach/cfd-test-problem-suite

# Install python 3.7
RUN apt-get install -y software-properties-common &&\
    add-apt-repository ppa:deadsnakes/ppa &&\
    apt-get update &&\
    apt-get install -y build-essential python3-pip python3.7 python3.7-dev swig &&\
    python3.7 -m pip install --upgrade pip &&\
    python3.7 -m pip install cython setuptools wheel

# Install benchmark dependencies.
COPY ./requirements.txt ./requirements_eob.txt

RUN python3.7 -m pip install -r requirements_eob.txt

COPY expensiveoptimbenchmark ./expensiveoptimbenchmark

RUN ln -s ./dockerCall.sh ./evaluate.sh