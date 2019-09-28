#!/usr/bin/env python
from __future__ import print_function
import argparse
import math
import pandas as pd
import numpy as np
import copy
from collections import Counter,deque
import operator
import itertools
import random
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



def Random_Forest(train,valid,test,heuristic):
    data=train
    valid_data=valid
    test_data=test
    data.columns=listofcolumns
    valid_data.columns=listofcolumns
    test_data.columns=listofcolumns
    print('---------------------------------------')
    datad=pd.concat([data,valid_data])
#     print(datad.shape)
    X=datad.iloc[:,[i for i in range(500)]].values
    y=datad.iloc[:,500].values.tolist()
    test_X=test_data.iloc[:,[i for i in range(500)]].values
    test_y=test_data.iloc[:,500].values

#adapted from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    n_estimators = [500,750,1000]
    max_features = ['auto']
    max_depth = [5,10,15,20]
    max_depth.append(None)
    grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}
 
    print('Begin Grid Search')
    GS = GridSearchCV(estimator=RandomForestClassifier(random_state=0, criterion=heuristic),param_grid=grid,cv=5)
    GS.fit(X, y)
    print(GS.best_params_)
#    p=GS.predict(test_X)
#    print("Accuracy for Random Forest on CV data: ",accuracy_score(test_y,p))
#    print('Only best param')
    GS_grid = {'n_estimators': GS.best_params_['n_estimators'],
               'max_features': GS.best_params_['max_features'],
               'max_depth': GS.best_params_['max_depth']
               }
    testGS=RandomForestClassifier(random_state=0,max_features=GS_grid['max_features'], n_estimators= GS_grid['n_estimators'], max_depth=GS_grid['max_depth'], criterion=heuristic)
    X=datad.iloc[:,[i for i in range(500)]].values
    y=datad.iloc[:,500].values
    testGS.fit(X,y)
    p=testGS.predict(test_X)
    print("Accuracy for test data on Random Forest using GridsearchCV: ",accuracy_score(test_y,p))

class Node:
    def __init__(self, attribute='',subset=pd.DataFrame(),heuristic_value=0.5,depth=0):
        self.attribute=attribute
        self.subset=subset
        self.heuristic=heuristic_value
        self.depth=depth
        self.zerochild=None
        self.onechild=None
        self.majority=-1

def printTreeDetailed(root,tab=0):
    if root==None:
        print('Null Node!')
        return
    if root.attribute=='' or root.attribute==None:
       for t in range(tab):
           print('     ',end='')
       purity,x=npPure(root.subset)
       subset=root.subset
       classvalues=subset[:,-1]
       c=Counter(classvalues)
       zeroes=c[0]
       ones=c[1]
       print('|isPure={0} zeroes{1} ones {2} Majority class={3} Depth={4}'.format(purity,zeroes,ones,root.majority, root.depth))
       return
    else:
       for t in range(tab):
           print('     ',end='')
       subset=root.subset
       classvalues=subset[:,-1]
       c=Counter(classvalues)
       zeroes=c[0]
       ones=c[1]
       print('|Branch on {0}=0, Depth {1}, zeroes={2}, ones={3}'.format(root.attribute,root.depth,zeroes,ones))
       printTreeDetailed(root.zerochild,tab+1)
       for t in range(tab):
           print('     ',end='')
       subset=root.subset
       classvalues=subset[:,-1]
       c=Counter(classvalues)
       zeroes=c[0]
       ones=c[1]
       print('|Branch on {0}=1, Depth {1}, zeroes={2}, ones={3}'.format(root.attribute,root.depth,zeroes,ones))
       printTreeDetailed(root.onechild,tab+1)
def printTree(root,tab=0):
    if root==None:
        print('Null Node!')
        return
    if root.attribute=='' or root.attribute==None:
       print('|={0}'.format(root.majority),end='')
       return
    else:
       print()
       for t in range(tab):
           print('-',end='')
       print('|{0} = 0'.format(root.attribute),end='')
       printTree(root.zerochild,tab+1)
       print()
       for t in range(tab):
           print('-',end='')
       print('|{0} = 1'.format(root.attribute),end='')
       printTree(root.onechild,tab+1)
def npPure(subset):
    classvalues=subset[:,-1]
    c=Counter(classvalues)
    zeroes=c[0]
    ones=c[1]
    maj=1
    if zeroes>ones:
        maj=0
    elif zeroes==ones:
        #maj=random.choice([0,1])
        maj=1
    return len(c.keys())==1,maj
  
  
def np_query(subset,colnum,val):
  return subset[np.where(subset[:,colnum] == val)]

def npNode(data,current_impurity,depth):
    col_names=listofcolumns
    denom=len(data)
    gain_list=[None]*len(col_names)
    for i,col in enumerate(col_names):
        s=np_query(data,i,1)
        one_leaf=self_entropy_np(s)
        zero_leaf=self_entropy_np(np_query(data,i,0))
        numer=len(s)
        fraction_one=numer/(denom*1.0)
        fraction_zero=1-fraction_one
        gain=current_impurity-fraction_one*one_leaf-fraction_zero*zero_leaf
        gain_list[i]=gain
#    print(gain_list)
    col_index=gain_list.index(max(gain_list))
    n=Node(col_names[col_index],data,current_impurity-gain_list[col_index],depth)
    #take care of majority class
    return n,col_index
def npNodev(data,current_impurity,depth):
    col_names=listofcolumns
    denom=len(data)
    gain_list=[None]*len(col_names)
    for i,col in enumerate(col_names):
        s=np_query(data,i,1)
        one_leaf=self_var_np(s)
        zero_leaf=self_var_np(np_query(data,i,0))
        numer=len(s)
        fraction_one=numer/(denom*1.0)
        fraction_zero=1-fraction_one
        gain=current_impurity-fraction_one*one_leaf-fraction_zero*zero_leaf
        gain_list[i]=gain
#    print(gain_list)
    col_index=gain_list.index(max(gain_list))
    n=Node(col_names[col_index],data,current_impurity-gain_list[col_index],depth)
    #take care of majority class
    return n,col_index 
  
def self_entropy_np(dataframe):
    #find column
    entropy=0
    if len(dataframe)==0:
        return entropy
    data_list=dataframe[:,-1]
    #data_list=dataframe['PlayTennis'].tolist()
    numer=np.bincount(data_list)[0]#count of the first bin
    denom=len(data_list)
    fraction=numer/(denom*1.0)
    if fraction!=0 and fraction!=1:
        entropy=(-1)*fraction*math.log(fraction,2)-((1-fraction)*math.log((1-fraction),2))
    #print(entropy)
    return entropy
  
  
def self_var_np(dataframe):
    variance=0
    if len(dataframe)==0:
        return variance
    data_list=dataframe[:,-1]
    numer=np.bincount(data_list)[0]#count of the first bin
    denom=len(data_list)
    if denom-numer==0 or numer==0:
        return variance
    variance=(numer)*(denom-numer)/(denom*denom*1.0)
    return variance  

def Construct_np(data,current_heuristic,depth):
    #Stop conditions->
    pure,majority=npPure(data) 
#     if len(data)==0:
#         pure,majority=isPure(data)
   
    if pure or len(data)==0:
        n=Node(None,data,current_heuristic,depth)
        n.majority=majority
        return n
    else:
        n,ind=npNode(data,current_heuristic,depth)
        n.majority=-1
        zero_data=np_query(data,ind,0)
        n.zerochild=Construct_np(zero_data,self_entropy_np(zero_data),depth+1 )
        one_data=np_query(data,ind,1)
        n.onechild=Construct_np(one_data,self_entropy_np(one_data),depth+1 )
    return n
  
def Construct_var(data,depthbound,current_heuristic,depth):
    #Stop conditions->
    pure,majority=npPure(data) 
#     if len(data)==0:
#         pure,majority=isPure(data)
   
    if pure or len(data)==0:
        n=Node(None,data,current_heuristic,depth)
        n.majority=majority

        return n
    else:
        n,ind=npNodev(data,current_heuristic,depth)
        n.majority=-1
#         n.set_zo()
        zero_data=np_query(data,ind,0)
        n.zerochild=Construct_var(zero_data,self_var_np(zero_data),depth+1 )
        one_data=np_query(data,ind,1)
        n.onechild=Construct_var(one_data,self_var_np(one_data),depth+1 )
    return n
  
def np_predictedValue(root,instance):
    if root.attribute=='' or root.attribute==None:
        return root.majority
    else:
#         index=col_names.index(root.attribute)
#         child=instance[index]
        child=instance[colToNum(root.attribute)]
        if child==0:
            #print('branch on 0')
            return np_predictedValue(root.zerochild,instance)
        else:
            #print('branch on 1')
            return np_predictedValue(root.onechild,instance)
def np_Accuracy(root,valid_set):
    data_A=valid_set
#    clist=data_A.columns.tolist()
#     clist=collectNodes(root)
    predictions=np.array([np_predictedValue(root,row) for row in data_A])
    actual=valid_set[:,-1]
    acc=(predictions==actual)
    return acc.sum()/(acc.size*1.0)
          
def collectNodes(root):
    """collects inner nodes of the tree"""
    innernodesBFS=[]
    queue=deque()
    queue.append(root)
    while(len(queue)!=0):
        #for n in queue: print(n.attribute)
        temp=queue.popleft()
        if temp.attribute==None:
            pass
        else:
            innernodesBFS.append(temp)
            #print(temp.attribute)
            if temp.onechild.attribute==None:
                queue.append(temp.zerochild)
                #go down zerochild only
            elif temp.zerochild.attribute==None:
                queue.append(temp.onechild)
            else:
                queue.append(temp.zerochild)
                queue.append(temp.onechild)
    return innernodesBFS

def copyLeafNode(destination,source):
    destination.subset=source.subset
    destination.attribute=source.attribute
    destination.zerochild=source.zerochild
    destination.onechild=source.onechild
    destination.majority=source.majority
    return destination    

def np_innerNodeToLeaf(root):
    n=root
    nsubset=np.concatenate([n.onechild.subset,n.zerochild.subset])
    b,maj=npPure(nsubset)
    n.attribute=None
    n.subset=nsubset
    n.zerochild=None
    n.onechild=None
    n.majority=maj
    return n
  
def np_pruneRE(root):
    newroot=copy.deepcopy(root)
    previous_A=np_Accuracy(newroot,valid_data)
    newrootlist=collectNodes(newroot)
    flag=True
    while(flag):
        #templist=collectNodes(temproot)
        newrootlist.reverse()
        templist=copy.deepcopy(newrootlist)
        end_t=len(templist)-1
        end_n=len(newrootlist)-1
        accuracies=[]
        app=accuracies.append
        for i in range(end_t):
            att=templist[i].attribute
            n=copy.deepcopy(templist[i])
            
            n=np_innerNodeToLeaf(n)
            templist[i]=copyLeafNode(templist[i],n)
            app(np_Accuracy(templist[end_t],valid_data))
#             print(i,' Attribute {0} trimmed accuracy={1}'.format(att,acc))
            templist[i]=copyLeafNode(templist[i],newrootlist[i])
                   
        index, value = max(enumerate(accuracies), key=operator.itemgetter(1))
#         print('Max of accuracies is  ',value)
        t=copy.deepcopy(newrootlist[index])
        #keep the old value 
        t=copyLeafNode(t,newrootlist[index])
        newrootlist[index]=np_innerNodeToLeaf(newrootlist[index])
        #print('Candidate node is  ',t.attribute,'at ',index)    
        new_A=np_Accuracy(newrootlist[end_n],valid_data)
        #print('New Accuracy is',new_A)
        newroot=newrootlist[end_n]
        #print('Previous roundaccuracy=',previous_A)
        #test if new_A>=previous_A
        if new_A <= previous_A:#does not continue through
            newrootlist[index]=copyLeafNode(newrootlist[index],t)
            flag=False
        else:
            previous_A=new_A
            newrootlist=collectNodes(newroot)
    return newroot


def prune_depth(root,dmax):
    """pass a copy of root in the first call"""
    #merge any childnode that has depth of dmax-1
    #traverse tree until it is a leaf with depth<=dmax or you find dmax
    if root.attribute==None or root.attribute=='':
        return
    if root.depth>=dmax:
        root=np_innerNodeToLeaf(root)
        return
    if root.majority>=0:
        return    
    else:
        prune_depth(root.zerochild,dmax)
        prune_depth(root.onechild,dmax)
    return root

def numToExcelCol(n):
    col = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        col = chr(65 + r) + col
    return col
def colToNum(col):
    n=0
    if len(col)>1:
      n=(ord(col[0])-65+1)*26+ord(col[1])-65
    else:
      n=ord(col[0])-65
    return n

parser = argparse.ArgumentParser(description='Description here')
   
parser.add_argument('-p', '--pathprefix',default='hw1_data/all_data/', 
    help="path directory to csv data")
parser.add_argument('-c', '--clauses', type=int,default='hw1_data/all_data/', required=True, 
    help="clauses:300,500,1000,1500,1800")
parser.add_argument('-d', '--examples',type=int, default='100', required=True, 
    help="examples: 100,1000, 5000")
parser.add_argument('-e','--heuristic',type=int, required=True, 
                    help="0: for entropy 1: for variance/gini")
parser.add_argument('-o','--option', type=int, required=True, 
                    help="0: No pruning "+
                    "1: Reduced-error pruning " +
                    "2: Depth-based pruning "+
                    "3: Random Forest")
parser.add_argument('-t','--printtree', type=int, required=True, 
                    help="0: no printing 1: print trees")

args = parser.parse_args()

    
if __name__=='__main__':
#     print(args.clauses, args.examples, args.heuristic, args.option, args.printtree)
    
    pathprefix=args.pathprefix
    sets=['train','test','valid']    
    clauses=args.clauses
    examples=args.examples
    tree_decision=args.option
    h_decision=args.heuristic
    heuristic='entropy'
    printtree=args.printtree
    
#     pathprefix='/content/drive/My Drive/Colab Notebooks/All_data/all_data/'
#     sets=['train','test','valid']    
#     clauses=300
#     examples=100
#     tree_decision=1
#     h_decision=0
#     heuristic='entropy'
#     printtree=0
    
    d_read=pd.read_csv(pathprefix+'train'+'_c'+str(clauses)+'_d'+str(examples)+'.csv',header=None)
    v_read=pd.read_csv(pathprefix+'valid'+'_c'+str(clauses)+'_d'+str(examples)+'.csv',header=None)
    t_read=pd.read_csv(pathprefix+'test'+'_c'+str(clauses)+'_d'+str(examples)+'.csv',header=None)
    data_df=pd.DataFrame(d_read)
    valid_data_df=pd.DataFrame(v_read)
    test_data_df=pd.DataFrame(t_read)
    listofcolumns=[ 'ZZ' for i in range(len(data_df.columns)-1)]
    #Assign string column names
    for i in range(len(listofcolumns)):
        listofcolumns[i]=numToExcelCol(i+1)
    listofcolumns.append('Class')
    data_df.columns=listofcolumns
    valid_data_df.columns=listofcolumns
    test_data_df.columns=listofcolumns
    
    
    if h_decision==1:
        heuristic='gini'
    if tree_decision==3:
        Random_Forest(data_df,valid_data_df,test_data_df,heuristic)
    else: 
        data=data_df.to_numpy()
        valid_data=valid_data_df.to_numpy()
        test_data=test_data_df.to_numpy()
        listofcolumns=[ 'ZZ' for i in range(len(data[0])-1)]
        #Assign string column names
        for i in range(len(listofcolumns)):
            listofcolumns[i]=numToExcelCol(i+1)        
        if h_decision==0:
            print('Entropy Heuristic')
            root=Construct_np(data,self_entropy_np(data),0)
            if printtree==1:
                print('pre pruned tree--------------------------')
                printTreeDetailed(root)
            print('pre prune accuracy vs validation data',np_Accuracy(root,valid_data))
            print('pre prune accuracy vs test data',np_Accuracy(root,test_data))
            if tree_decision==1:
                newrootRE=copy.deepcopy(root)
                newrootRE=np_pruneRE(newrootRE)
                print('RE pruning')
                print('post prune accuracy vs validation data',np_Accuracy(newrootRE,valid_data))
                print('post prune accuracy vs test data',np_Accuracy(newrootRE,test_data))
                if printtree==1:
                    print('post prune tree Reduced Error-----------------')
                    printTreeDetailed(newrootRE)
            elif tree_decision==2:
                newroot=copy.deepcopy(root)
                newroot=prune_depth(newroot,5)
                newroot10=copy.deepcopy(root)
                newroot10=prune_depth(newroot10,10)
                newroot15=copy.deepcopy(root)
                newroot15=prune_depth(newroot15,15)
                newroot20=copy.deepcopy(root)
                newroot20=prune_depth(newroot20,20)
                newroot50=copy.deepcopy(root)
                newroot50=prune_depth(newroot50,50)
                newroot100=copy.deepcopy(root)
                newroot100=prune_depth(newroot100,100)
                
                print('Entropy: Depth based pruning')
                print('post prune n=max depth')
                print('post prune 5 accuracy vs validation data',np_Accuracy(newroot,valid_data))
                print('post prune 5 accuracy vs test data',np_Accuracy(newroot,test_data))
                print('post prune 10 accuracy vs validation data',np_Accuracy(newroot10,valid_data))
                print('post prune 10 accuracy vs test data',np_Accuracy(newroot10,test_data))
                print('post prune 15 accuracy vs validation data',np_Accuracy(newroot15,valid_data))
                print('post prune 15 accuracy vs test data',np_Accuracy(newroot15,test_data))
                print('post prune 20 accuracy vs validation data',np_Accuracy(newroot20,valid_data))
                print('post prune 20 accuracy vs test data',np_Accuracy(newroot20,test_data))
                print('post prune 50 accuracy vs validation data',np_Accuracy(newroot50,valid_data))
                print('post prune 50 accuracy vs test data',np_Accuracy(newroot50,test_data))
                print('post prune 100 accuracy vs validation data',np_Accuracy(newroot100,valid_data))
                print('post prune 100 accuracy vs test data',np_Accuracy(newroot100,test_data))
                                
                
                if printtree==1:
                    print('post prune tree Depth =5-----------------')
                    printTreeDetailed(newroot)
                    print('post prune tree Depth =10-----------------')
                    printTreeDetailed(newroot10)
                    print('post prune tree Depth =15-----------------')
                    printTreeDetailed(newroot15)
                    print('post prune tree Depth =20-----------------')
                    printTreeDetailed(newroot20)
                    print('post prune tree Depth =50-----------------')
                    printTreeDetailed(newroot50)
                    print('post prune tree Depth =100-----------------')
                    printTreeDetailed(newroot100)
        else:
            rootv=Construct_var(data,self_var_np(data),0)
            print('Variance Heuristic')
            print('pre prune accuracy vs validation data',np_Accuracy(rootv,valid_data))
            print('pre prune accuracy vs test data',np_Accuracy(rootv,test_data))
            if printtree==1:
                print('pre pruned tree--------------------------')
                printTreeDetailed(rootv)
            if tree_decision==1:
                newrootv=copy.deepcopy(rootv)
                newrootv=np_pruneRE(newrootv)
                print('RE pruning')
                print('post prune accuracy vs validation data',np_Accuracy(newrootv,valid_data))
                print('post prune accuracy vs test data',np_Accuracy(newrootv,test_data))
                if printtree==1:
                    print('post prune tree Reduced Error-----------------')
                    printTreeDetailed(newrootv)
            elif tree_decision==2:
                newroot=copy.deepcopy(rootv)
                newroot=prune_depth(newroot,5)
                newroot10=copy.deepcopy(rootv)
                newroot10=prune_depth(newroot10,10)
                newroot15=copy.deepcopy(rootv)
                newroot15=prune_depth(newroot15,15)
                newroot20=copy.deepcopy(rootv)
                newroot20=prune_depth(newroot20,20)
                newroot50=copy.deepcopy(rootv)
                newroot50=prune_depth(newroot50,50)
                newroot100=copy.deepcopy(rootv)
                newroot100=prune_depth(newroot100,100)
                
                print('Variance: Depth based pruning')
                print('post prune n=max depth')
                print('post prune 5 accuracy vs validation data',np_Accuracy(newroot,valid_data))
                print('post prune 5 accuracy vs test data',np_Accuracy(newroot,test_data))
                print('post prune 10 accuracy vs validation data',np_Accuracy(newroot10,valid_data))
                print('post prune 10 accuracy vs test data',np_Accuracy(newroot10,test_data))
                print('post prune 15 accuracy vs validation data',np_Accuracy(newroot15,valid_data))
                print('post prune 15 accuracy vs test data',np_Accuracy(newroot15,test_data))
                print('post prune 20 accuracy vs validation data',np_Accuracy(newroot20,valid_data))
                print('post prune 20 accuracy vs test data',np_Accuracy(newroot20,test_data))
                print('post prune 50 accuracy vs validation data',np_Accuracy(newroot50,valid_data))
                print('post prune 50 accuracy vs test data',np_Accuracy(newroot50,test_data))
                print('post prune 100 accuracy vs validation data',np_Accuracy(newroot100,valid_data))
                print('post prune 100 accuracy vs test data',np_Accuracy(newroot100,test_data))
                
                
                if printtree==1:
                    print('post prune tree Depth =5-----------------')
                    printTreeDetailed(newroot)
                    print('post prune tree Depth =10-----------------')
                    printTreeDetailed(newroot10)
                    print('post prune tree Depth =15-----------------')
                    printTreeDetailed(newroot15)
                    print('post prune tree Depth =20-----------------')
                    printTreeDetailed(newroot20)
                    print('post prune tree Depth =50-----------------')
                    printTreeDetailed(newroot50)
                    print('post prune tree Depth =100-----------------')
                    printTreeDetailed(newroot100)