Bootstrap: docker
From: frehbach/cfd-test-problem-suite
Stage: spython-base

%environment
export PYTHONPATH=$PYTHONPATH:/home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/:/home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/data
export WM_MPLIB=SYSTEMOPENMPI

%files
./requirements.txt ./requirements_eob.txt
expensiveoptimbenchmark /home/openfoam/expensiveoptimbenchmark
./expensiveoptimbenchmark/problems/ESP2/dockerCall.py /home/openfoam/cfd-test-problem-suite/
./expensiveoptimbenchmark/problems/ESP2/dockerCall.sh /home/openfoam/cfd-test-problem-suite/
./expensiveoptimbenchmark/problems/ESP2/createBafflesDict2.py /home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/ESP/createBafflesDict2.py
./expensiveoptimbenchmark/problems/ESP2/evaluateSimulation2.py /home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/ESP/evaluateSimulation2.py

%post

apt-get install -y software-properties-common &&\
add-apt-repository ppa:deadsnakes/ppa &&\
apt-get update &&\
apt-get install -y build-essential python3-pip python3-tk python3.7 python3.7-dev swig &&\
python3.7 -m pip install --upgrade pip &&\
python3.7 -m pip install --upgrade cython setuptools wheel
python3.7 -m pip install -r requirements_eob.txt

# bash ./expensiveoptimbenchmark/problems/ESP2/patch.sh
ln -s /home/openfoam/cfd-test-problem-suite/dockerCall.sh /evaluate.sh

chmod -R o+rwX /home/openfoam

%runscript
exec /bin/bash -l -c "$@"
%startscript
exec /bin/bash -l -c "$@"
