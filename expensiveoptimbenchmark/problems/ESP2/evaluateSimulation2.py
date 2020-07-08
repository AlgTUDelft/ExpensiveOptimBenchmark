#!/usr/bin/env python
# coding: utf8

# ver 0.2 TBB modified system calls to ensure bash shell

import os
import sys
import csv
import ntpath
import time
from subprocess import call

## Accept decimal information string via command line from R
baffleStr = sys.argv[1]
# Modification: Do not replace `,` with ''. It is no longer the case that every element has their own index.
# baffleStr = baffleStr.replace(',', '')

## Create a copy of the base simulation configuration
os.system('/bin/bash -c "cp -r Exeter_CFD_Problems/ESP/baseCase Exeter_CFD_Problems/ESP/foamWorkingDir"')

# Parse the baffle configuration vector into an OpenFOAM readable format
os.system("python2.7 Exeter_CFD_Problems/ESP/createBafflesDict2.py " + baffleStr)

call('/bin/bash -c "cd Exeter_CFD_Problems/ESP/foamWorkingDir\ncreateBaffles -overwrite"',shell=True,stdout = open(os.devnull,'wb'))
call('/bin/bash -c "cd Exeter_CFD_Problems/ESP/foamWorkingDir\nsimpleFoam"',shell=True,stdout = open(os.devnull,'wb'))
call('/bin/bash -c "cd Exeter_CFD_Problems/ESP/foamWorkingDir\npostProcess -func sampleDict -latestTime"',shell=True,stdout = open(os.devnull,'wb'))

sys.exit(call("python2.7 Exeter_CFD_Problems/ESP/postProcessConsole.py",shell=True))
