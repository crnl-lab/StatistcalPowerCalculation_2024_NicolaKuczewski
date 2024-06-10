"""
Created on 14/06/2018

@author: Nicola
"""

from __future__ import division
from multiprocessing import Pool, TimeoutError
import time
from scipy import *
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#import pandas as pd 
#import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest


def generate_sample(N, desired_mean, desired_std, epsilon):
    # Generate a list with random numbers
    if desired_std==0:
        sample= np.full(shape=sample_size, fill_value=smpl_mean)
        return(sample)
  
    
    
    while True:
        # Generate a list of N numbers between 0 and 1
        num_list = np.random.rand(N)
        
        # Scale the list to desired mean and standard deviation
        scaled_list = num_list * desired_std / np.std(num_list)
        scaled_list = scaled_list - np.mean(scaled_list) + desired_mean
     
    
        # Clip values to ensure they are between 0 and 1
        scaled_list = np.clip(scaled_list, 0, 1)
     
        
        # Check if mean and standard deviation are within tolerance
        if abs(np.mean(scaled_list) - desired_mean) / desired_mean <= epsilon and \
           abs(np.std(scaled_list) - desired_std) / desired_std <= epsilon:
            break
    
    return scaled_list




def CI95(data):
    CI=st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    return[CI[0],CI[1]],np.mean(data)-CI[0]
   
    
def Welch (suc1,suc2,animals_sucess1,animals_sucess2,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided’'
    p = st.ttest_ind(animals_sucess1,animals_sucess2,equal_var=False,alternative=alter)[1]
    return (p)
def MannWhitney (suc1,suc2,animals_sucess1,animals_sucess2,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    if sum(abs(suc1-suc2))==0:return (1)
    p = st.mannwhitneyu (animals_sucess1,animals_sucess2, alternative=alter)[1]
    return (p)
    
def Ttest1(success1, animals_sucess1,chance,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    p=st.ttest_1samp(animals_sucess1,chance,alternative=alter)[1]
    return(p)

def Ttest2(success1,success2,animals_sucess1,animals_sucess2,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    p=st.ttest_ind(animals_sucess1,animals_sucess2,alternative=alter)[1]
    return(p)
    
def  Wilcoxon1(success1, animals_sucess1,chance,tst): 
    
    data=animals_sucess1-chance
    if np.sum (np.abs(sum(data)))==0: return(1)
    p=st.wilcoxon(data)[1]
    return(p) 
    
def proportion1(success1, animals_sucess1,chance,tst):
  
    if tst==0:
      alter='larger'
    else:
      alter='two-sided'
    succes=np.sum(success1)
    total= len(success1)
    p=proportions_ztest(succes,total,value=chance,alternative=alter)[1]
    return(p)
               

def proportion2(suc1,suc2,animals_sucess1,animals_sucess2,tst): 
    if tst==0:
        alter='larger'
    else:
        alter='two-sided'
    success_Gp1=np.sum(suc1)
    success_Gp2=np.sum(suc2)
    total1=len(suc1)
    total2=len(suc2)
    p=proportions_ztest([success_Gp1,success_Gp2],[total1,total2],alternative=alter)[1]
    return(p)
               

def fisher(suc1,suc2,animals_sucess1,animals_sucess2,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    success_Gp1=np.sum(suc1)
    success_Gp2=np.sum(suc2)
    missed1=len(suc1)-success_Gp1
    missed2=len(suc2)-success_Gp2
    oddratio,p=st.fisher_exact([[success_Gp1, missed1], [success_Gp2, missed2]],alternative=alter)
    return(p)
    
    




   
def run_simulation(
    groups=1, # number of experimental groups (if 1 the succes probability is compared to the chache level)
    tests=Ttest1, # Ttest1 (compared to the chache level),Ttest2(compare 2 groups),proportion1,proportion2,Wilcoxon1    nWhitney    ,Welch" ,'Permutation','MannWhitney'
    trials=8,  # trial repetitions for  each subject
    success_Gp1=50, # % average probability of succes of trained group
    std1=10,# standard deviation of succes  between subjects in  trained group
    success_Gp2=30, #% average probability of succes of control group  or chancelevel when only 1 experimental group
    std2=25,  # standard deviation of succes  between subjects in control group
    sample_size1=4,
    sample_size2=6,
    alpha_risk=0.05,    
    iterations=5000,
    repetitions=5, # number or simulations repetition for calculate the average power and 95%CI
    tolerance=0.05, #tolerance error in generated sample mean and std
    ):
        
    
    if success_Gp1<success_Gp2 :
        raise ValueError("Sucecs rate of group 1 should be ≥ of sucecs rate of groupe 2")
    prob1=success_Gp1/100
    std1/=100
    prob2=success_Gp2/100
    std2/=100
    
    
    if prob1==prob2: 
        alternative=1
    else:
        alternative =0
    sample1=generate_sample(sample_size1,prob1,std1,tolerance)
    # sample1=[0.82372467, 0.70869149, 0.9017045,  0.52705134, 0.60300111, 0.63375287]
    print('Groupe 1 probability of success:',sample1)
    print ('Average probability of groupe 1:',np.mean(sample1))
    print ('STD of groupe 1:',np.std(sample1))
    print()
    if groups==2:
        sample2=generate_sample(sample_size2,prob2,std2,tolerance)
        # sample2=[0.62003418, 0.23798336, 0.44689089, 0.56752667, 0.58428336, 0.54627834]
        print('Groupe 2 probability of success:',sample2)
        print (' Average probability of groupe 2',np.mean(sample2))
        print (' STD of groupe 2',np.std(sample2))
        print()
    power=[]# ratio of false negatives
    for rep in range(repetitions):
        fn=0#false negatives 
        fp=0#false positive when no difference betwen prob1 and prob2
        for it in range(iterations): 
            success1=[] #  succes cumulated over the animals
            animals_sucess1=[]# succes rate by animal
            for panimal in sample1:
                anim_suc=[]
                for trial in range(trials):
                    suc1=np.random.binomial(1,panimal,size=1)
                    success1.append(suc1[0])
                    anim_suc.append(suc1[0])
                animals_sucess1.append(sum(anim_suc)/trials)
            
            success1=np.array(success1)
            
            if groups==1:
                if tests in [Ttest2,proportion2,fisher,Welch,MannWhitney]: raise Exception ('Wrong test for compareson of one group with the chance level')
                pval=tests(success1, animals_sucess1,prob2,alternative) 
                
            elif groups==2:
                if tests in [Ttest1,proportion1]: raise Exception ('Wrong test for compareson between  two groups')
                success2=[]# all succes cumulated by group
                animals_sucess2=[]# succes rate by animal
                for panimal in sample2:
                    anim_suc=[]
                    for trial in range(trials):
                        suc2=np.random.binomial(1,panimal,size=1)
                        success2.append(suc2[0])
                        anim_suc.append(suc2[0])
                    animals_sucess2.append(sum(anim_suc)/trials)
                success2=np.array(success2)
                
                
                pval=tests(success1,success2,animals_sucess1,animals_sucess2,alternative)  
              
                            
            if pval>alpha_risk:
                fn+=1  
            else:
                fp+=1
        
        if prob1==prob2:
            print('% Type I error:',fp/iterations*100)
            power.append(fp/iterations*100)
        else :
            print('Power:',100-fn/iterations*100)
        
            power.append(100-fn/iterations*100)
        
    if prob1==prob2:
        print('% Type I error mean:',np.mean(power))
        print('% Type I error 95 CI:',(CI95(power)))
    else:
        print()
        print('Power mean:',np.mean(power))
        print('Power 95 CI:',(CI95(power)))
      
    
if __name__=="__main__":
    run_simulation()