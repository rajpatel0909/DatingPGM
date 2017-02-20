# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "rajpu&mihir"
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
    df = pd.read_csv("SpeedDating_discrete02.csv");
    file1 = open("cmd.txt");
    file2 = file.read(file1);
    model = eval(file2)
    #Finding CPDS
    pe = ParameterEstimator(model, df) 
     #CPD FOR NODE: LIKE
    cpd = pe.state_counts('match')
    cpd = cpd.transpose() #Helps when summing
    #print cpd
    #Baap Code to get the probabilities from the frquencies.
    cpd_prob = cpd.div(cpd.sum(axis=1), axis=0) 
    #Remove the divide b 0 error
    cpd_prob = cpd_prob.fillna(0)
    print cpd_prob
