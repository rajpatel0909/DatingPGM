# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "rajpu"
__date__ = "$Feb 20, 2017 10:29:20 AM$"

import openpyxl
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination

if __name__ == "__main__":
    print "Hello World"
    
    #df = pd.read_csv("C:/MyStuff/SEM2/AML/Project1/tes.csv");
    df = pd.read_csv("C:/MyStuff/SEM2/AML/Project1/SpeedDating_discrete02.csv");

    model=BayesianModel([('gender', 'attr2_1'),('gender', 'sinc2_1'),('gender', 'intel2_1'),
                         ('gender', 'fun2_1'),('gender', 'amb2_1'),('gender', 'shar2_1'),
                         ('gender', 'goal'),('gender', 'condtn'),
    ('condtn', 'dec'),('condtn', 'attr1_1'),('condtn', 'sinc1_1'),('condtn', 'intel1_1'),('condtn', 'fun1_1'),
    ('condtn', 'amb1_1'),('condtn', 'shar1_1'),
    ('age', 'goal'),('age', 'attr_o'),('age', 'intel_o'),('age', 'fun_o'),('age', 'sinc_o'),('age', 'condtn'),
    ('age', 'attr2_1'),('age', 'sinc2_1'),('age', 'intel2_1'),('age', 'fun2_1'),('age', 'amb2_1'),('age', 'shar2_1'),
    ('samerace', 'int_corr'),
    ('race_o', 'samerace'),
    ('race', 'samerace'),
    ('race', 'condtn'),
    ('imprace', 'condtn'),
    ('imprelig', 'condtn'),
    ('int_corr', 'dec'),
    ('goal', 'dec'),
    ('met', 'dec'),
    ('dec', 'match'),
    ('dec_o', 'match'),
    ('career_c', 'sinc_o'),
    ('career_c', 'intel_o'),
    ('field_cd', 'sinc_o'),
    ('field_cd', 'intel_o'),
    ('date', 'fun_o'),
    ('age_o','pf_o_att'),('age_o','pf_o_sin'),('age_o','pf_o_int'),('age_o','pf_o_fun'),('age_o','pf_o_amb'),
    ('age_o','pf_o_sha'),('age_o','attr'),('age_o','sinc'),('age_o','intel'),('age_o','fun'),
    ('pf_o_att','dec_o'),('pf_o_sin','dec_o'),('pf_o_int','dec_o'),('pf_o_fun','dec_o'),('pf_o_amb','dec_o'),
    ('pf_o_sha','dec_o'),
    ('attr1_1','dec'),('sinc1_1','dec'),('intel1_1','dec'),('fun1_1','dec'),('amb1_1','dec'),
    ('shar1_1','dec'),
    ('attr','like'),('sinc','like'),('intel','like'),('fun','like'),
    ('like','dec'),
    ('imprace','imprelig'),
    ('go_out','date')
    ])
    pe = ParameterEstimator(model, df)  
    cpd = pe.state_counts('match')
    print(cpd)        

    print (model.active_trail_nodes("dec"))
    
    model.fit(df, MaximumLikelihoodEstimator)
