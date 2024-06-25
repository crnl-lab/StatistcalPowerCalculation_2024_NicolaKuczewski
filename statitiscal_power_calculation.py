"""
Created on 14/06/2018

@author: Nicola Kuczewski and Samuel Garcia
"""
from scipy import *
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportions_ztest


def generate_sample(sample_size, desired_mean, desired_std, epsilon, rng=None):
    # Generate a list  of twith random numbers
    if desired_std==0:
        sample= np.full(shape=sample_size, fill_value=desired_mean)
        return(sample)  
    while True:
        # Generate a list of N numbers between 0 and 1
        # num_list = np.random.rand(sample_size)
        if rng is None:
            rng = np.random.default_rng(seed=None)
        num_list = rng.uniform(low=0, high=1, size=sample_size)

        
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
    CI=scipy.stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=scipy.stats.sem(data)) 
    return[CI[0],CI[1]],np.mean(data)-CI[0]
   
    
def Welch (suc1,suc2,animals_sucess1,animals_sucess2,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided’'
    p = scipy.stats.ttest_ind(animals_sucess1,animals_sucess2,equal_var=False,alternative=alter)[1]
    return (p)

def MannWhitney (suc1,suc2,animals_sucess1,animals_sucess2,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    if sum(abs(suc1-suc2))==0:return (1)
    p = scipy.stats.mannwhitneyu (animals_sucess1,animals_sucess2, alternative=alter)[1]
    return (p)
    
def Ttest1(success1, animals_sucess1,chance,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    p=scipy.stats.ttest_1samp(animals_sucess1,chance,alternative=alter)[1]
    return(p)

def Ttest2(success1,success2,animals_sucess1,animals_sucess2,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    p=scipy.stats.ttest_ind(animals_sucess1,animals_sucess2,alternative=alter)[1]
    return(p)
    
def  Wilcoxon1(success1, animals_sucess1,chance,tst): 
    
    data=animals_sucess1-chance
    if np.sum (np.abs(sum(data)))==0: return(1)
    p=scipy.stats.wilcoxon(data)[1]
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
    success_rate_grp1=np.sum(suc1)
    success_rate_grp2=np.sum(suc2)
    total1=len(suc1)
    total2=len(suc2)
    p=proportions_ztest([success_rate_grp1,success_rate_grp2],[total1,total2],alternative=alter)[1]
    return(p)
               

def fisher(suc1,suc2,animals_sucess1,animals_sucess2,tst):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    success_rate_grp1=np.sum(suc1)
    success_rate_grp2=np.sum(suc2)
    missed1=len(suc1)-success_rate_grp1
    missed2=len(suc2)-success_rate_grp2
    oddratio, p = scipy.stats.fisher_exact([[success_rate_grp1, missed1], [success_rate_grp2, missed2]],alternative=alter)
    return(p)
    
    


test_methods = {
    'Ttest2': Ttest2,
    'proportion2': proportion2,
    'fisher': fisher,
    'Welch': Welch,
    'MannWhitney': MannWhitney,
    'Ttest1': Ttest1,
    'proportion1': proportion1,
}

   
def run_simulation(
    mode='one-group',#  'two-group'
    method_name='Ttest1', # Ttest1 (compared to the chance level),Ttest2(compare 2 num_group),proportion1,proportion2,Wilcoxon1    nWhitney    ,Welch" ,'Permutation','MannWhitney'
    


    num_trial=8,  # trial num_simulation for  each subject
    
    success_rate_grp1=50, # % average probability of succes of trained group
    std_grp1=10,# standard deviation of succes  between subjects in  trained group
    sample_size_grp1=4,

    chance_level_grp1=50.,

    success_rate_grp2=30, #% average probability of succes of control group  or chancelevel when only 1 experimental group
    std_grp2=25,  # standard deviation of succes  between subjects in control group
    sample_size_grp2=6,

    alpha_risk=0.05,    
    num_iteration=5000,
    num_simulation=5, # number or simulations repetition for calculate the average power and 95%CI
    tolerance=0.05, #tolerance error in generated sample mean and std
    seed=None,
    ):
    """
    This script uses a Monte Carlo simulation to calculate the statistical power of behavioral studies that evaluate  success rates. 
    It determines how the modification of different  parameters of the  bheavoural and analytical protocol affects statistical power
    

    Parameters
    ----------

    mode: str, 'one-group' | 'two-group'
        If 'one-group' the succes probability is compared to the chance level
        If 'two-group' the succes of group1 is compared to group2
    method_name : str , default 'Ttest1'
        If num_group == 1, compared to the chance level:
            * 'Ttest1', 'proportion1'
        If num_group == 2, compare 2 num_group:
            * 'Ttest2', 'proportion2', 'Wilcoxon1', 'nWhitney', 'Welch' , 'Permutation','MannWhitney'
    num_trial : int, default 8
        Trial num_simulation for each subject
    success_rate_grp1 : float, default 50.
        % average probability of succes of trained group
    std_grp1 : float, default 10.,
        Standard deviation of succes  between subjects in  trained group
    sample_size_grp1 : int, default 4
        Sample size group 1
    chance_level_grp1: float, default 50.
        When mode='one-group', this is the chance level.
    success_rate_grp2 : float, default 30.
        % average probability of succes of control group
        Used when mode=='two-group'
    std_grp2 : float, default 25.
        Standard deviation of succes  between subjects in control group
        Used when mode=='two-group'
    sample_size_grp2 : int, default 6
        Sample size group 2
        Used when mode=='two-group'
    alpha_risk : float default 0.05
        Risk alpha
    num_iteration : int, default 5000
        Number of iteration to computation Monte-Carlo simulation
    num_simulation : int, default 5
        Number or simulations repetition for calculate the average power and 95%CI
    tolerance : float, default 0.05
        Tolerance error in generated sample mean and std
    seed: Nont or int
        The generation seed
    """
    
    if method_name not in test_methods:
        raise ValueError(f"method_name={method_name} must be in {list(test_methods.keys())}")
    test_method_function = test_methods[method_name]

    if mode not in ('one-group', 'two-group'):
        raise ValueError("mode mus be 'one-group', 'two-group'")
    
    if success_rate_grp1<success_rate_grp2 :
        raise ValueError("Sucecs rate of group 1 should be ≥ of sucecs rate of groupe 2")
    
    prob1 = success_rate_grp1 / 100
    std_grp1 = std_grp1 / 100.
    if mode == 'one-group':
        prob2 = chance_level_grp1 / 100.
    elif mode == 'two-group':
        prob2 = success_rate_grp2 / 100.
        std_grp2 = std_grp2 / 100.
    
    if prob1==prob2:
        #'two-sided'
        alternative = 1
    else:
        # 'greater'
        alternative = 0


    rng = np.random.default_rng(seed=seed)

    sample1=generate_sample(sample_size_grp1,prob1,std_grp1,tolerance, rng=rng)
    # sample1=[0.82372467, 0.70869149, 0.9017045,  0.52705134, 0.60300111, 0.63375287]
    print('Groupe 1 probability of success:',sample1)
    print ('Average probability of groupe 1:',np.mean(sample1))
    print ('STD of groupe 1:',np.std(sample1))
    print()
    # if num_group==2:
    if mode == 'two-group':
        sample2=generate_sample(sample_size_grp2,prob2,std_grp2,tolerance, rng=rng)
        # sample2=[0.62003418, 0.23798336, 0.44689089, 0.56752667, 0.58428336, 0.54627834]
        print('Groupe 2 probability of success:',sample2)
        print (' Average probability of groupe 2',np.mean(sample2))
        print (' STD of groupe 2',np.std(sample2))
        print()
    power=[]# ratio of false negatives
    for rep in range(num_simulation):
        fn=0#false negatives 
        fp=0#false positive when no difference betwen prob1 and prob2
        for it in range(num_iteration): 
            success1=[] #  succes cumulated over the animals
            animals_sucess1=[]# succes rate by animal
            for panimal in sample1:
                anim_suc=[]
                for trial in range(num_trial):
                    suc1 = rng.binomial(1, p=panimal, size=1)
                    success1.append(suc1[0])
                    anim_suc.append(suc1[0])
                animals_sucess1.append(sum(anim_suc)/num_trial)
            
            success1=np.array(success1)
            
            # if num_group==1:
            if mode == 'one-group':
                if method_name not in ['Ttest1','proportion1']:
                    raise Exception ('Wrong test for compareson of one group with the chance level')
                pval=test_method_function(success1, animals_sucess1,prob2,alternative) 
                
            # elif num_group==2:
            elif mode == 'two-group':
                if method_name not in ['Ttest2','proportion2','fisher','Welch','MannWhitney']:
                    raise Exception ('Wrong test for compareson between  two num_group')
                
                success2=[]# all succes cumulated by group
                animals_sucess2=[]# succes rate by animal
                for panimal in sample2:
                    anim_suc=[]
                    for trial in range(num_trial):
                        suc2 = rng.binomial(1, p=panimal, size=1)
                        success2.append(suc2[0])
                        anim_suc.append(suc2[0])
                    animals_sucess2.append(sum(anim_suc)/num_trial)
                success2=np.array(success2)
                
                pval = test_method_function(success1,success2,animals_sucess1,animals_sucess2,alternative)  

            if prob1==prob2 and pval<alpha_risk:
                fp+=1

            if prob1>prob2 and pval>alpha_risk:
                fn+=1


        if prob1==prob2:
            print('% Type I error:', fp/num_iteration * 100)
            power.append(fp/num_iteration*100)
        else :
            print('Power:',100-fn/num_iteration*100)
            power.append(100-fn/num_iteration*100)
        
    if prob1==prob2:
        print('% Type I error mean:',np.mean(power))
        print('% Type I error 95 CI:',(CI95(power)))
    else:
        print()
        print('Power mean:',np.mean(power))
        print('Power 95 CI:',(CI95(power)))
      
    
if __name__=="__main__":
    run_simulation(
        mode="one-group",
        method_name='Ttest1',
        
        num_trial=8,

        success_rate_grp1=70.,
        std_grp1=5.,
        sample_size_grp1=4,

        chance_level_grp1=50.,

        success_rate_grp2=70.,
        std_grp2=10.,
        sample_size_grp2=6,

        alpha_risk=0.05,
        num_iteration=5000,
        num_simulation=5,
        tolerance=0.05,
        seed=2205,
        # seed=None
    )
