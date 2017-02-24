# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
#from traitlets.config.application import catch_config_error
from numba.typing.enumdecl import infer

__author__ = "rajpu&mihir"
__date__ = "$Feb 20, 2017 10:29:20 AM$"

#import openpyxl
import multiprocessing
import findInfer
import time
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
from pgmpy.sampling import BayesianModelSampling

def findInfer1(infer):
    print(infer.query(['match'],evidence={'race':1,'race_o':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1,'imprace':5,'imprelig':5,'gender':0,'like':1,'like_o':1}) ['match'])
    return

def findInfer2(infer):
    print(infer.query(['goal'],evidence={'gender':1,'dec':1,'attr2_1':0,'sinc2_1':0,'intel2_1':0,'fun2_1':0,'amb2_1':0,'shar2_1':1,'race':1,'race_o':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':0,'imprace':5,'imprelig':5}) ['goal'])
    return

def findInfer3(infer):
    print(infer.query(['age'],evidence={'attr2_1':0,'sinc2_1':0,'intel2_1':0,'fun2_1':0,'amb2_1':0,'shar2_1':1,'attr_o':2,'sinc_o':2,'intel_o':2,'fun_o':2,'like_o':2,'match':1}) ['age'])
    return

def findInfer4(infer):
    print(infer.query(['gender'],evidence={'attr2_1':0,'sinc2_1':0,'intel2_1':0,'fun2_1':0,'amb2_1':0,'shar2_1':1,'imprace':1,'imprelig':1,'goal':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1}) ['gender'])
    return

def findInfer5(infer):
    print(infer.query(['dec'],evidence={'attr':0,'sinc':0,'intel':0,'fun':0,'like':0,'imprace':1,'imprelig':1, 'condtn':1,'goal':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1,'met':1}) ['dec'])
    return

if __name__ == '__main__':
    print "Hello World"
    
    #df = pd.read_csv("C:/MyStuff/SEM2/AML/Project1/tes.csv");
    df = pd.read_csv("newData.csv");
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
    
    #findInfer.findInfer1(infer)
    
    p1 = multiprocessing.Process(target=findInfer1, args=(infer,))
    p2 = multiprocessing.Process(target=findInfer2, args=(infer,))
    p3 = multiprocessing.Process(target=findInfer3, args=(infer,))
    p4 = multiprocessing.Process(target=findInfer4, args=(infer,))
    p5 = multiprocessing.Process(target=findInfer5, args=(infer,))
    
    
#     inference = BayesianModelSampling(model)
#     print("inference done")
#     var = inference.forward_sample(2, str)
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    #print(infer.query(['match'],evidence={'race':1,'race_o':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1,'imprace':5,'imprelig':5,'gender':0,'like':1,'like_o':1}) ['match'])
    #print(infer.query(['goal'],evidence={'gender':1,'dec':1,'attr2_1':0,'sinc2_1':0,'intel2_1':0,'fun2_1':0,'amb2_1':0,'shar2_1':1,'race':1,'race_o':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':0,'imprace':5,'imprelig':5}) ['goal'])
    #print(infer.query(['age'], evidence={'match':1}) ['age'])
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
    
