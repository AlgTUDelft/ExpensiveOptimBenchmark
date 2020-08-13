"""
================================================================================
Docker Call Script - CFD Test Problems
================================================================================
:Author:
    Steven Daniels <S.Daniels@exeter.ac.uk>
    Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
    Frederik Rehbach <frederik.rehbach@th-koeln.de>
:Date:
    24 July, 2019
:Copyright:
   Copyright (c)  Steven Daniels and Alma Rahat, University of Exeter, 2018
:File:
   dockerCall.py
"""

import sys, time
import numpy as np
import os
from subprocess import call
from shutil import rmtree

# To avoid tkinter errors.
import matplotlib
matplotlib.use('agg')

seed = 1435
np.random.seed(seed)

# import problem classes
import Exeter_CFD_Problems as TestProblems

allArgs = sys.argv

if __name__=='__main__':
    if '-v' in sys.argv:
        verbose = True
    else:
        verbose = False
    if '-p' in sys.argv:
        problem_name = sys.argv[sys.argv.index('-p') + 1]
    else:
        raise ValueError('No valid problem name was defined')
    sys.argv = sys.argv[:1] # this is required for PyFoam to work correctly
    if problem_name == 'PitzDaily':
        # set up directories.
        settings = {
            'source_case': 'Exeter_CFD_Problems/data/PitzDaily/case_fine/',
            'case_path': 'Exeter_CFD_Problems/data/PitzDaily/case_single/',
            'boundary_files': ['Exeter_CFD_Problems/data/PitzDaily/boundary.csv'],
            'fixed_points_files': ['Exeter_CFD_Problems/data/PitzDaily/fixed.csv']
        }
        # instantiate the problem object
        prob = TestProblems.PitzDaily(settings)
        # get the lower and upper bounds
        lb, ub = prob.get_decision_boundary()
        # last call argument should contain decision vector
        x = np.fromstring(allArgs[-1], sep=',')
        res = prob.evaluate(x, verbose=verbose)
        sys.exit(res)
    elif problem_name == 'KaplanDuct':
        print('Demonstration of the KaplanDuct test problem.')
        # set up directories.
        settings = {
            'source_case': 'Exeter_CFD_Problems/data/KaplanDuct/case_fine/',
            'case_path': 'Exeter_CFD_Problems/data/KaplanDuct/case_single/',
            'boundary_files': ['Exeter_CFD_Problems/data/KaplanDuct/boundary_1stspline.csv', \
                                'Exeter_CFD_Problems/data/KaplanDuct/boundary_2ndspline.csv'],
            'fixed_points_files': ['Exeter_CFD_Problems/data/KaplanDuct/fixed_1.csv',\
                                'Exeter_CFD_Problems/data/KaplanDuct/fixed_2.csv']
        }
        # instantiate the problem object
        prob = TestProblems.KaplanDuct(settings)
        lb, ub = prob.get_decision_boundary()
        x = np.fromstring(allArgs[-1], sep=',')
        res = prob.evaluate(x, verbose=verbose)
        sys.exit(res)
    elif problem_name == 'HeatExchanger':
        print('Demonstration of the HeatExchanger test problem.')
        # set up directories.
        settings = {
            'source_case': 'Exeter_CFD_Problems/data/HeatExchanger/heat_exchange/',
            'case_path': 'Exeter_CFD_Problems/data/HeatExchanger/case_multi/'
        }
        # instantiate the problem object
        prob = TestProblems.HeatExchanger(settings)
        lb, ub = prob.get_decision_boundary()
        x = np.fromstring(allArgs[-1], sep=',')
        res = prob.evaluate(x, verbose=verbose)
        sys.exit(res)
    elif problem_name == 'ESP':
        print('Demonstration of the ESP test problem.')
        res = os.system('python Exeter_CFD_Problems/ESP/evaluateSimulation.py ' + allArgs[-1])
        rmtree('Exeter_CFD_Problems/ESP/foamWorkingDir')
        sys.exit(res)
    elif problem_name == 'ESP2':
        print('Demonstration of the ESP variant two test problem.')
        res = os.system('python Exeter_CFD_Problems/ESP/evaluateSimulation2.py ' + allArgs[-1])
        rmtree('Exeter_CFD_Problems/ESP/foamWorkingDir')
        sys.exit(res)
    elif problem_name == 'ESP3':
        print('Demonstration of the ESP variant three test problem.')
        res = os.system('python Exeter_CFD_Problems/ESP/evaluateSimulation3.py ' + allArgs[-1])
        rmtree('Exeter_CFD_Problems/ESP/foamWorkingDir')
        sys.exit(res)
    elif problem_name == 'ESP4':
        print('Demonstration of the ESP variant three test problem.')
        res = os.system('python Exeter_CFD_Problems/ESP/evaluateSimulation4.py ' + allArgs[-1])
        rmtree('Exeter_CFD_Problems/ESP/foamWorkingDir')
        sys.exit(res)