Bootstrap: docker
From: frehbach/cfd-test-problem-suite
Stage: spython-base

%files
./requirements.txt ./requirements_eob.txt
expensiveoptimbenchmark ./expensiveoptimbenchmark
%post

apt-get install -y software-properties-common &&\
add-apt-repository ppa:deadsnakes/ppa &&\
apt-get update &&\
apt-get install -y build-essential python3-pip python3.7 python3.7-dev swig &&\
python3.7 -m pip install --upgrade pip &&\
python3.7 -m pip install --upgrade cython setuptools wheel
python3.7 -m pip install -r requirements_eob.txt

bash ./expensiveoptimbenchmark/problems/ESP2/patch.sh
ln -s ./dockerCall.sh ./evaluate.sh

%runscript
exec /bin/bash -l -c "$@"
%startscript
exec /bin/bash -l -c "$@"
