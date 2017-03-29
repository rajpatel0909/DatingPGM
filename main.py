# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
#from traitlets.config.application import catch_config_error
from numba.typing.enumdecl import infer
import neuralNet

__author__ = "raj&mihir"
__date__ = "$Feb 20, 2017 10:29:20 AM$"

#import openpyxl
import multiprocessing
import time
import getNodes
import getLinks
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pgmpy
#from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models.BayesianModel import BayesianModel
#from pgmpy.inference import GibbsSampling
#from pgmpy.inference import Sampling 
from pgmpy.sampling import GibbsSampling
from pgmpy.sampling import HamiltonianMC as HMC, LeapFrog, GradLogPDFGaussian
from pgmpy.factors.continuous import JointGaussianDistribution as JGD

from scipy.stats import norm

def findMatch(infer):
    #print "Given the partners and hostzs decisions, Find Probability distribution for match between them"
    print "Given the partners and hosts decisions, Find Probability distribution for match between them\n", (infer.query(['match'],evidence={'race':1,'race_o':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1,'imprace':5,'imprelig':5,'gender':0,'like':1,'like_o':1}) ['match'])
    return

def findGoal(infer):
    #print "Given the hosts gender, condition, decision and preferences, Find Probability distribution for goal of host"
    print "Given the hosts gender, condition, decision and preferences, Find Probability distribution for goal of host\n", (infer.query(['goal'],evidence={'samerace':1,'race_o':2,'race':2,'condtn':1,'int_corr':2,'dec':0,'attr1_1':1,'sinc1_1':1,'intel1_1':1,'fun1_1':1,'amb1_1':0,'shar1_1':0,'imprace':1,'imprelig':1,'met':1,'gender':0,'age':2,'attr2_1':1,'sinc2_1':0,'intel2_1':1,'fun2_1':1,'amb2_1':0,'shar2_1':0,'dec_o':0,'match':0,'attr_o':2,'sinc_o':2,'intel_o':2,'fun_o':2,'like_o':2,'career_c':1,'field_cd':1,'date':5,'go_out':1,'pf_o_att':2,'pf_o_sin':1,'pf_o_int':1,'pf_o_fun':1,'pf_o_amb':0,'pf_o_sha':0,'age_o':3,'attr':2,'sinc':2,'intel':2,'fun':2,'like':2}) ['goal'])
    return

def findAge(infer):
    #print "Given the hosts preferences, age, condition and Find Probability distribution for gender of age"
    print "Given the hosts decision, gender, preferences and rating, Find Probability distribution for age of host\n", (infer.query(['age'],evidence={'attr2_1':0,'sinc2_1':0,'intel2_1':0,'fun2_1':0,'amb2_1':0,'shar2_1':1,'attr_o':2,'sinc_o':2,'intel_o':2,'fun_o':2,'like_o':2,'match':1}) ['age'])
    return

def findGender(infer):
    #print "Given the hosts decision, gender, preferences and rating, Find Probability distribution for gender of host"
    print "Given the hosts preferences, age, condition and Find Probability distribution for gender of host\n", (infer.query(['gender'],evidence={'samerace':1,'race_o':2,'race':2,'condtn':1,'int_corr':2,'dec':0,'attr1_1':1,'sinc1_1':1,'intel1_1':1,'fun1_1':1,'amb1_1':0,'shar1_1':0,'imprace':1,'imprelig':1,'met':1,'goal':1,'age':2,'attr2_1':1,'sinc2_1':0,'intel2_1':1,'fun2_1':1,'amb2_1':0,'shar2_1':0,'dec_o':0,'match':0,'attr_o':2,'sinc_o':2,'intel_o':2,'fun_o':2,'like_o':2,'career_c':1,'field_cd':1,'date':5,'go_out':1,'pf_o_att':2,'pf_o_sin':1,'pf_o_int':1,'pf_o_fun':1,'pf_o_amb':0,'pf_o_sha':0,'age_o':3,'attr':2,'sinc':2,'intel':2,'fun':2,'like':2}) ['gender'])
    return

def findDecn(infer, nnw):
    x = [1,1,6,7,6,7,7,5,7,7,7,9,7,8,7,1,8]
    i = findIntcorr(nnw, x)
    print(infer.query(['dec'],evidence={'int_corr':2,'attr':0,'sinc':0,'intel':0,'fun':0,'like':0,'imprace':1,'imprelig':1, 'condtn':1,'goal':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1,'met':1}) ['dec'])
    return

def findDec(infer):
    #print "Given the hosts preferences and ratings, Find Probability distribution for decision of host"
    print "Given the hosts preferences and ratings, Find Probability distribution for decision of host\n", (infer.query(['dec'],evidence={'attr':0,'sinc':0,'intel':0,'fun':0,'like':0,'imprace':1,'imprelig':1, 'condtn':1,'goal':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1,'met':1}) ['dec'])
    return

def findDeco(infer):
    #print "Given the partners preferences and ratings, Find Probability distribution for decision of partner"
    print "Given the partners preferences and ratings, Find Probability distribution for decision of partner\n", (infer.query(['dec_o'],evidence={'pf_o_att':0,'pf_o_sin':0,'pf_o_int':0,'pf_o_fun':0,'like_o':0,'career_c':1,'field_cd':1, 'go_out':1,'date':1,'attr_o':0,'sinc_o':0,'intel_o':0,'fun_o':0,'age':1}) ['dec_o'])
    return

def findIntcorr(nnw, x):
    return neuralNet.predictNN(nnw, x)
    
if __name__ == '__main__':
    #print "Hello World"
    
    print "Reading data"
    
    df = pd.read_csv("newData.csv");
    #
    print "Calculating Mean and Entropy"
    mean = {}
    entropy = {}
    for i in range(df.shape[1]):
        entropy[list(df[[i]])[0]] = -1 * np.sum(norm.logpdf(df[[i]]))/df.shape[0]
        mean[list(df[[i]])[0]] = np.mean(df[[i]])
        
    
    print "Training Neural Network for continous variables"
    #uncomment this
    nnw = neuralNet.trainNeuralNetwork()
    
    
    file1 = open("cmd.txt");
    file2 = file.read(file1);
    #model = eval(file2)
    
    print "Creating Bayesian Model"
    model = BayesianModel(getLinks.getLinksOfNodes())
    #print("Making parents dictionary")
    parents = getLinks.getParents()
   
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
            #print(node, e)
            tempException1 = 0
            
    # #         
    print("Calculating tabular CPDs")
    
  
    
    
    parentsCardList = {}
    for node in nodes:
        if parents.has_key(node):
            tempParentCardList = []
            for parent in parents[node]:
                tempParentCardList.append(len(cpds[parent]))
            parentsCardList[node] = tempParentCardList
        
    print("Adding Tabular CPDs to model")    
    
    for node in nodes:
        try:
            if parents.has_key(node):
                model.add_cpds(TabularCPD(node, variableCard[node], cpds[node], parents[node], parentsCardList[node]))
            else:
                model.add_cpds(TabularCPD(node, variableCard[node], cpds[node]))
        except Exception as e:
            tempException2 = 0
            
    #print("Tabular cpds added to model")
    
    print "Creating samples using Bayesian model forward Sampling"
    
    inference = BayesianModelSampling(model)
    normalSamples = inference.forward_sample(size=5000, return_type='dataframe')
    #print "length ", normalSamples.shape
    
    print "Some of the samples are as follows"
    print normalSamples[1:2]
    print " "
    print "Calculating relative entropies between different Sampling models"
    smean = {}
    sentropy = {}
    for i in range(normalSamples.shape[1]):
        sentropy[list(normalSamples[[i]])[0]] = -1 * np.sum(norm.logpdf(normalSamples[[i]]))/normalSamples.shape[0]
        smean[list(normalSamples[[i]])[0]] = np.mean(normalSamples[[i]])
    
    relEntropy = {}
    
    print " "
    #print "Variable        Mean        Entropy        Relative Entropy"
    print "{:<10} {:<10} {:<10} {:<10}".format('Variable','Mean','Entropy','Relative Entropy')
    
    for k in sentropy.keys():
        relEntropy[k] = sentropy[k] - entropy[k]
        print "{:<10} {:<10} {:<10} {:<10}".format(k,format(round(mean[k][0],3)),format(round(entropy[k],3)),format(round(relEntropy[k],3)))
        #print k, "\t\t", format(round(mean[k][0],3)), "\t", format(round(entropy[k],3)),"\t", format(round(relEntropy[k],3))
        
    print " "
    

    
    infer = VariableElimination(model)
    
    #infer.map_query((['dec'],evidence={'attr':0,'sinc':0,'intel':0,'fun':0,'like':0,'imprace':1,'imprelig':1, 'condtn':1,'goal':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1,'met':1}) ['dec']))
    #infer = BeliefPropagation(model)
    print "Calculating Map Queries\n"
    print "Given the ratings by partner, Decisions of partner and host and Characteristics of Partner and host, Find Most likely values of Preferences of Partner"
    print(infer.map_query(variables = ['pf_o_att','pf_o_sin','pf_o_int','pf_o_fun','pf_o_amb','pf_o_sha'], evidence={'samerace':1,'race_o':2,'race':2,'condtn':0,'int_corr':2,'dec':0,'attr1_1':1,'sinc1_1':1,'intel1_1':1,'fun1_1':1,'amb1_1':0,'shar1_1':0,'imprace':1,'imprelig':1,'met':1,'goal':1,'gender':0,'age':2,'attr2_1':1,'sinc2_1':0,'intel2_1':1,'fun2_1':1,'amb2_1':0,'shar2_1':0,'dec_o':0,'match':0,'attr_o':0,'sinc_o':0,'intel_o':0,'fun_o':0,'like_o':0,'career_c':1,'field_cd':1,'date':5,'go_out':1,'age_o':3,'attr':2,'sinc':2,'intel':2,'fun':2,'like':2}))
    
    
    print "\nGiven the partners preferences, Decisions of partner and host and Characteristics of Partner and host, Find Most likely values of Ratings by Partner"
    print(infer.map_query(variables = ['attr_o','sinc_o','intel_o','fun_o','like_o'], evidence={'samerace':1,'race_o':2,'race':2,'condtn':1,'int_corr':2,'dec':0,'attr1_1':1,'sinc1_1':1,'intel1_1':1,'fun1_1':1,'amb1_1':0,'shar1_1':0,'imprace':1,'imprelig':1,'met':1,'goal':1,'gender':0,'age':2,'attr2_1':1,'sinc2_1':0,'intel2_1':1,'fun2_1':1,'amb2_1':0,'shar2_1':0,'dec_o':0,'match':0,'career_c':1,'field_cd':1,'date':5,'go_out':1,'pf_o_att':2,'pf_o_sin':1,'pf_o_int':1,'pf_o_fun':1,'pf_o_amb':0,'pf_o_sha':0,'age_o':3,'attr':2,'sinc':2,'intel':2,'fun':2,'like':2}))
    
    print "\nGiven the Rating by host, Decisions of partner and host and Characteristics of Partner and host, Find Most likely values of Preferences of Host"
    print(infer.map_query(variables = ['attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1','shar1_1'], evidence={'samerace':1,'race_o':2,'race':2,'condtn':1,'int_corr':2,'dec':0,'imprace':1,'imprelig':1,'met':1,'goal':1,'gender':0,'age':2,'attr2_1':1,'sinc2_1':0,'intel2_1':1,'fun2_1':1,'amb2_1':0,'shar2_1':0,'dec_o':0,'match':0,'attr_o':2,'sinc_o':2,'intel_o':2,'fun_o':2,'like_o':2,'career_c':1,'field_cd':1,'date':5,'go_out':1,'pf_o_att':2,'pf_o_sin':1,'pf_o_int':1,'pf_o_fun':1,'pf_o_amb':0,'pf_o_sha':0,'age_o':3,'attr':2,'sinc':2,'intel':2,'fun':2,'like':2}))
    
    print "\nGiven the hosts preferences, Decisions of partner and host and Characteristics of Partner and host, Find Most likely values of Ratings of Host"
    print(infer.map_query(variables = ['attr2_1','sinc2_1','intel2_1','fun2_1','amb2_1','shar2_1'], evidence={'samerace':1,'race_o':2,'race':2,'condtn':1,'int_corr':2,'dec':0,'attr1_1':1,'sinc1_1':1,'intel1_1':1,'fun1_1':1,'amb1_1':0,'shar1_1':0,'imprace':1,'imprelig':1,'met':1,'goal':1,'gender':0,'age':2,'dec_o':0,'match':0,'attr_o':2,'sinc_o':2,'intel_o':2,'fun_o':2,'like_o':2,'career_c':1,'field_cd':1,'date':5,'go_out':1,'pf_o_att':2,'pf_o_sin':1,'pf_o_int':1,'pf_o_fun':1,'pf_o_amb':0,'pf_o_sha':0,'age_o':3,'attr':2,'sinc':2,'intel':2,'fun':2,'like':2}))
    
    print "\nGiven the hosts and partners preferences, Decisions of partner and host and Characteristics of Partner and host, Find Most likely values of Ratings of Host"
    print(infer.map_query(variables = ['attr','sinc','intel','fun','like'], evidence={'samerace':1,'race_o':2,'race':2,'condtn':1,'int_corr':2,'dec':0,'attr1_1':1,'sinc1_1':1,'intel1_1':1,'fun1_1':1,'amb1_1':0,'shar1_1':0,'imprace':1,'imprelig':1,'met':1,'goal':1,'gender':0,'age':2,'attr2_1':1,'sinc2_1':0,'intel2_1':1,'fun2_1':1,'amb2_1':0,'shar2_1':0,'dec_o':0,'match':0,'attr_o':2,'sinc_o':2,'intel_o':2,'fun_o':2,'like_o':2,'career_c':1,'field_cd':1,'date':5,'go_out':1,'pf_o_att':2,'pf_o_sin':1,'pf_o_int':1,'pf_o_fun':1,'pf_o_amb':0,'pf_o_sha':0,'age_o':3}))
    
    #print(infer.max_marginal(variables = ['dec'],evidence={'attr':0,'sinc':0,'intel':0,'fun':0,'like':0,'imprace':1,'imprelig':1, 'condtn':1,'goal':1,'attr1_1':0,'sinc1_1':0,'intel1_1':0,'fun1_1':0,'amb1_1':0,'shar1_1':1,'met':1}))
    
    #print("model fitted")
    #print(model.check_model())
    
    #findInfer.findInfer1(infer)
    
    print "\nCalculating Probability Queries\n"
    
    findDec(infer)
    findDeco(infer)
    findGender(infer)
    findAge(infer)
    findMatch(infer)
    findGoal(infer)
    
    
    
