# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
#from traitlets.config.application import catch_config_error

__author__ = "rajpu&mihir"
__date__ = "$Feb 20, 2017 10:29:20 AM$"

#import openpyxl
import getNodes
import getLinks
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


print "Hello World"

#df = pd.read_csv("C:/MyStuff/SEM2/AML/Project1/tes.csv");
df = pd.read_csv("C:/Users/rajpu/workspace/DatingPGM/SpeedDating_discrete02.csv");
file1 = open("cmd.txt");
file2 = file.read(file1);
#model = eval(file2)
model = BayesianModel(getLinks.getLinksOfNodes())
print("Making parents dictionary")
parents = getLinks.getParents()
print("Parents Dictionary made")
#model = BayesianModel([('goal','dec'),('met','dec'),('condtn','dec'),('int_corr','dec'),('attr1_1','dec'),('like','dec'),('sinc1_1','dec'),('amb1_1','dec')])

#model = BayesianModel([('dec','match'),('dec','match')])

#print(model)
#Finding CPDS
pe = ParameterEstimator(model, df) 

nodes = getNodes.getNodesFromCSV();

cpds = {}
variableCard = {}



for node in nodes:
    try:
        cpd = pe.state_counts(node)
        cpd = cpd.transpose()
        cpd_prob = cpd.div(cpd.sum(axis=1), axis=0)
        cpd_prob = cpd_prob.fillna(1.0/cpd_prob.shape[1])
        cpds[node] = cpd_prob.transpose().values.tolist()
        variableCard[node] = cpd_prob.shape[1]
    except Exception as e:
        #nodes.remove(node)
        print(node, e)
        
# #         
print("cpds generated")

# test = pe.state_counts('match')
# test = test.transpose()
# test_prob = test.div(test.sum(axis=1), axis=0)
# test_prob = test_prob.fillna(0.5)
# print(test_prob)
# print((test_prob[0][0]))

#print(1 - cpds['fun_o'].transpose().sum(axis = 1))

print("Generating list pf variablecard of parents")
#making list of parents's variableCard


parentsCardList = {}
for node in nodes:
    if parents.has_key(node):
        tempParentCardList = []
        for parent in parents[node]:
            tempParentCardList.append(len(cpds[parent]))
        parentsCardList[node] = tempParentCardList
    
print("Generating and Adding Tabular cpd")    

for node in nodes:
    try:
        if parents.has_key(node):
            model.add_cpds(TabularCPD(node, variableCard[node], cpds[node], parents[node], parentsCardList[node]))
        else:
            model.add_cpds(TabularCPD(node, variableCard[node], cpds[node]))
    except Exception as e:
        print e
        
print("Tabular cpds added to model")
#print(cpds['fun_o'])
#temp_cpd = TabularCPD('age_o', variableCard['age_o'], cpds['age_o'])
#temp_cpd = TabularCPD('match', 2, cpds['match'], ['dec', 'dec_o'], [2, 2])
#model.add_cpds(temp_cpd)
#model.fit(df, MaximumLikelihoodEstimator)

infer = VariableElimination(model)

print("model fitted")
print(model.check_model())
#print (infer.query(['match'],evidence={'dec':0,'dec_o':0}) ['match'])
#mle = MaximumLikelihoodEstimator(model, df)
#print(mle.estimate_cpd('dec'))
# cpd = pe.state_counts('match')
# cpd = cpd.transpose() #Helps when summing
# #print cpd
# #Baap Code to get the probabilities from the frquencies.
# cpd_prob = cpd.div(cpd.sum(axis=1), axis=0) 
# #Remove the divide b 0 error
# cpd_prob = cpd_prob.fillna(0)
# print cpd_prob
