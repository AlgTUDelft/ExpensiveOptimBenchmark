Bootstrap: docker
From: frehbach/cfd-test-problem-suite
Stage: spython-base

%environment
export PATH=$PATH:/home/openfoam/platforms/linux64GccDPInt32Opt/bin
export PYTHONPATH=$PYTHONPATH:/home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/:/home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/data
export WM_MPLIB=SYSTEMOPENMPI
export JULIA_DEPOT_PATH=:/opt/julia

%files
./requirements.txt ./requirements_eob.txt
expensiveoptimbenchmark /home/openfoam/expensiveoptimbenchmark
./expensiveoptimbenchmark/problems/ESP2/dockerCall.py /home/openfoam/cfd-test-problem-suite/
./expensiveoptimbenchmark/problems/ESP2/dockerCall.sh /home/openfoam/cfd-test-problem-suite/
./expensiveoptimbenchmark/problems/ESP2/createBafflesDict2.py /home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/ESP/createBafflesDict2.py
./expensiveoptimbenchmark/problems/ESP2/evaluateSimulation2.py /home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/ESP/evaluateSimulation2.py
./expensiveoptimbenchmark/problems/ESP2/createBafflesDict3.py /home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/ESP/createBafflesDict3.py
./expensiveoptimbenchmark/problems/ESP2/evaluateSimulation3.py /home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/ESP/evaluateSimulation3.py
./expensiveoptimbenchmark/problems/ESP2/createBafflesDict4.py /home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/ESP/createBafflesDict4.py
./expensiveoptimbenchmark/problems/ESP2/evaluateSimulation4.py /home/openfoam/cfd-test-problem-suite/Exeter_CFD_Problems/ESP/evaluateSimulation4.py

%post
apt-get install -y software-properties-common &&\
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
python3.7 -m pip install -r requirements_eob.txt

# Install julia, for julia based DONEjl.
wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.0-linux-x86_64.tar.gz
tar -xvzf julia-1.5.0-linux-x86\_64.tar.gz
ln -s "$PWD"/julia-1.5.0/bin/julia /bin/
export JULIA_DEPOT_PATH=/opt/julia
julia -e 'using Pkg; ENV["PYTHON"]="python3.7"; pkg"add PyCall NLopt Distributions"'
chmod -R 645 /opt/julia
python3.7 -c "import julia; julia.install()"

# bash ./expensiveoptimbenchmark/problems/ESP2/patch.sh
ln -s /home/openfoam/cfd-test-problem-suite/dockerCall.sh /evaluate.sh

chmod -R o+rwX /home/openfoam

%runscript
exec /bin/bash -l -c "$@"
%startscript
exec /bin/bash -l -c "$@"
