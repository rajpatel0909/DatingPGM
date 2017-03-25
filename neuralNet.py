import numpy as np
import pandas as pd
from blaze.expr.expressions import shape


def NeuralNetwork():
    #trainX, trainY, testX, testY, ImageX, ImageY
    print "Neural Network"
    
    data = pd.read_csv('continousData.csv', sep=',',header=None)
    
    dataX = data.as_matrix(columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    dataY = data.as_matrix(columns = [17])
    trainX = dataX[0:4500]
    trainY = dataY[0:4500]
    validX = dataX[4501:5001]
    validY = dataY[4501:5001]
    testX = dataX[5001:]
    testY = dataY[5001:]
    
    print shape(trainX)
    print shape(trainY)
    print shape(validX)
    print shape(validY)
    print shape(testX)
    print shape(testY)
    
    
    eta2List = [0.03]
    maxCorrect = 0
    maxE1 = 0
    valmaxCorrect = 0
    valmaxE1 = 0
    testmaxCorrect = 0
    testmaxE1 = 0
    #imgmaxCorrect = 0
    #imgmaxE1 = 0
    maxE2 = 0
    count1 = 0
    count2 = 0
    allErrors = np.zeros(shape=(50,1))
    for eta in eta2List:
        #print "count2 = ",count2
        count2 += 1
        N = 1
        M = 12
        D = 17
        K = 11
        iterations = 4500
        w1 = np.random.randn(D,M)
        w2 = np.random.randn(M,K)
        b1 = np.ones(shape=(N,M))
        b2 = np.ones(shape=(N,K))
        for r in range(0,50):
            #print r
            for i in range(0,iterations):
                #print i
                x = trainX[(i*N):(i*N+N)]
                t = np.zeros(shape=(N,K))
                for j in range(0,N):
                    k = i*N + j
                    temp1 = int(trainY[k])
                    t[j][int(trainY[k])] = 1
                
                #layer1
                a1 = np.dot(x,w1) + b1
                z = 1/(1 + np.exp(-a1))
                
                #layer2
                a2 = np.dot(z,w2) + b2
                expa2 = np.exp(a2)
                y = expa2/(np.sum(expa2, axis=1).reshape(N,1))
                
                #layer2 Error
                delta2 = np.subtract(y,t)/N
                w2 = np.subtract(w2, np.multiply(eta, np.dot(np.transpose(z), delta2)))
                b2 = b2 - eta*delta2
                #layer1 Error
                delta1 = np.zeros(shape=(N,M))
                dw2 = np.dot(delta2, np.transpose(w2))
                for i in range(0,N):
                    temp = np.dot((z[i,:]),np.transpose(1-z[i,:]))
                    #temp = np.dot((a2[i,:]),np.transpose(1-a2[i,:]))
                    delta1[i,:] = np.multiply(temp,dw2[i,:])
                
                
                #allErrors[i*N:i*N+N,:] = -np.sum(np.multiply((y - t),(a2)), axis=1).reshape(N,1)
                w1 = np.subtract(w1, np.multiply(eta, np.dot(np.transpose(x), delta1)))
                b1 = b1 - eta*delta1
                
             
            
        #train predicting values
        
        yt = np.zeros(shape=(4500,11))
        for i in range(0,iterations):
            xt = trainX[(i*N):(i*N+N)]
            tt = np.zeros(shape=(N,K))
            for j in range(0,N):
                k = i*N + j
                tt[j][int(trainY[k])] = 1
            
            #layer1
            at1 = np.dot(xt,w1) + b1
            zt = 1/(1 + np.exp(-at1))
            
            #layer2
            at2 = np.dot(zt,w2) + b2
            expat2 = np.exp(at2)
            yt[i*N:i*N+N,:] = expat2/(np.sum(expat2, axis=1).reshape(N,1))
    
        predictedValues = np.zeros(shape=(4500,1))
        correct = 0
        wrong = 0
        for i in range(0,4500):
            preIndex = np.where(yt[i,:] == yt[i,:].max())[0]  
            predictedValues[i][0] = preIndex
            if preIndex == trainY[i]:
                correct += 1
            else:
                wrong += 1 
                
        if(maxCorrect < correct):
            maxCorrect = correct
            maxE1 = eta
            maxE2 = eta
            
        #valid prdicting values
        valyt = np.zeros(shape=(500,11))
        for i in range(0,500):
            xt = validX[(i*N):(i*N+N)]
            tt = np.zeros(shape=(N,K))
            for j in range(0,1):
                k = i*N + j
                temp1 = int(validY[k])
                tt[j][int(validY[k])] = 1
             
            #layer1
            at1 = np.dot(xt,w1) + b1
            zt = 1/(1 + np.exp(-at1))
             
            #layer2
            at2 = np.dot(zt,w2) + b2    
            expat2 = np.exp(at2)
            valyt[i*N:i*N+N,:] = expat2/(np.sum(expat2, axis=1).reshape(N,1))
     
        valpredictedValues = np.zeros(shape=(500,1))
        valcorrect = 0
        valwrong = 0
        for i in range(0,500):
            preIndex = np.where(valyt[i,:] == valyt[i,:].max())[0]  
            valpredictedValues[i][0] = preIndex
            if preIndex == validY[i]:
                valcorrect += 1
            else:
                valwrong += 1 
         
        allErrors[r,0] = valwrong/500
        if(valmaxCorrect < valcorrect):
            valmaxCorrect = valcorrect
            valmaxE1 = eta
                
                
        #test prdicting values
        testyt = np.zeros(shape=(566,11))
        for i in range(0,566):
            xt = testX[(i*N):(i*N+N)]
            tt = np.zeros(shape=(N,K))
            for j in range(0,1):
                k = i*N + j
                tt[j][int(testY[k])] = 1
            
            #layer1
            at1 = np.dot(xt,w1) + b1
            zt = 1/(1 + np.exp(-at1))
            
            #layer2
            at2 = np.dot(zt,w2) + b2    
            expat2 = np.exp(at2)
            testyt[i*N:i*N+N,:] = expat2/(np.sum(expat2, axis=1).reshape(N,1))
    
        testpredictedValues = np.zeros(shape=(566,1))
        testcorrect = 0
        testwrong = 0
        for i in range(0,566):
            preIndex = np.where(testyt[i,:] == testyt[i,:].max())[0]  
            testpredictedValues[i][0] = preIndex
            if preIndex == testY[i]:
                testcorrect += 1
            else:
                testwrong += 1 
            
        if(testmaxCorrect < testcorrect):
            testmaxCorrect = testcorrect
            testmaxE1 = eta 
            
        #USPS prdicting values
#         imgyt = np.zeros(shape=(20000,10))
#         for i in range(0,20):
#             xt = ImageX[(i*N):(i*N+N)]
#             tt = np.zeros(shape=(N,K))
#             for j in range(0,N):
#                 k = i*N + j
#                 tt[j][int(ImageY[k][0])] = 1
#             
#             #layer1
#             at1 = np.dot(xt,w1) + b1
#             zt = 1/(1 + np.exp(-at1))
#             
#             #layer2
#             at2 = np.dot(zt,w2) + b2    
#             expat2 = np.exp(at2)
#             imgyt[i*N:i*N+N,:] = expat2/(np.sum(expat2, axis=1).reshape(N,1))
#     
#         imgpredictedValues = np.zeros(shape=(20000,1))
#         imgcorrect = 0
#         imgwrong = 0
#         for i in range(0,20000):
#             preIndex = np.where(imgyt[i,:] == imgyt[i,:].max())[0]  
#             imgpredictedValues[i][0] = preIndex
#             if preIndex == ImageY[i]:
#                 imgcorrect += 1
#             else:
#                 imgwrong += 1 
#             
#         if(imgmaxCorrect < imgcorrect):
#             imgmaxCorrect = imgcorrect
#             imgmaxE1 = eta 
                
                
        print "Accuracy of Training Data ", (maxCorrect/4500)
        print "Accuracy of Test Data ", (testmaxCorrect/500)
        print "Accuracy of Valid Data ", (valmaxCorrect/566)
#         print "Accuracy of USPS Data ", (imgmaxCorrect/20000)
    
        """
        graphX = list(range(50))        
        plt.figure(1)
        plt.plot(graphX,allErrors)
        plt.xlabel("data points")
        plt.ylabel("errors")
        plt.title("change in error")
        plt.show()
        
        graphX = list(range(50000))        
        plt.figure(2)
        plt.plot(graphX, predictedValues,'r--', graphX, trainY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Singal Neural Network training data")
        plt.show()
        
        graphX = list(range(10000))        
        plt.figure(3)
        plt.plot(graphX, valpredictedValues,'r--', graphX, validY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Singal Neural Network test data")
        plt.show()
    
        graphX = list(range(10000))        
        plt.figure(4)
        plt.plot(graphX, testpredictedValues,'r--', graphX, testY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Singal Neural Network valid data")
        plt.show()
        
        graphX = list(range(20000))        
        plt.figure(5)
        plt.plot(graphX, imgpredictedValues,'r--', graphX, ImageY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Singal Neural Network USPS data")
        plt.show()
        """
        