"""
Created on 14/06/2018

@author: Nicola Kuczewski and Samuel Garcia
"""
from scipy import *
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings("ignore")




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
    
    return np.mean(data)-CI[0]
   
    
def Welch (suc1,suc2,animals_sucess1,animals_sucess2,tst,num_trial):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided’'
    p = scipy.stats.ttest_ind(animals_sucess1,animals_sucess2,equal_var=False,alternative=alter)[1]
    return (p)

def MannWhitney (suc1,suc2,animals_sucess1,animals_sucess2,tst,num_trial):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    if sum(abs(suc1-suc2))==0:return (1)
    p = scipy.stats.mannwhitneyu (animals_sucess1,animals_sucess2, alternative=alter)[1]
    return (p)
    
def Ttest1(success1, animals_sucess1,chance,tst,num_trial):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    p=scipy.stats.ttest_1samp(animals_sucess1,chance,alternative=alter)[1]
    return(p)

def Ttest2(success1,success2,animals_sucess1,animals_sucess2,tst,num_trial):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    p=scipy.stats.ttest_ind(animals_sucess1,animals_sucess2,alternative=alter)[1]
    return(p)
    
def  Wilcoxon1(success1, animals_sucess1,chance,tst,num_trial): 
    
    data=animals_sucess1-chance
    if np.sum (np.abs(sum(data)))==0: return(1)
    p=scipy.stats.wilcoxon(data)[1]
    return(p) 
    
def Proportion1(success1, animals_sucess1,chance,tst,num_trial):
  
    if tst==0:
      alter='larger'
    else:
      alter='two-sided'
    succes=np.sum(success1)
    total= len(success1)
    p=proportions_ztest(succes,total,value=chance,alternative=alter)[1]
    return(p)
               

def Proportion2(suc1,suc2,animals_sucess1,animals_sucess2,tst,num_trial): 
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
               

def Fisher(suc1,suc2,animals_sucess1,animals_sucess2,tst,num_trial):
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
    

def Multilevel_Mixed_Model(suc1,suc2,animals_success1,animals_success2,tst,num_trial):
     
    
    #create dataframe for LME
    n_total=len(animals_success1)+len(animals_success2)
    animal_id=[] 
    for animal in np.arange(n_total):
        animal_id=animal_id+[animal]*num_trial
        
    data = {
  'animal_id': animal_id,
  'group': ['GR1']*len(animals_success1)*num_trial + ['Gr2']*len(animals_success2)*num_trial,  
  'trial': list(range(1, num_trial+1))*n_total,  
  'success':list(suc1)+list(suc2) 
  }
      
    df = pd.DataFrame(data)
    
    df['group'] = df['group'].astype('category')
   
     
     # Fit a mixed-effects model with group as the fixed effect and animal_id as the random effect
    model = smf.mixedlm("success ~ group", df, groups=df["animal_id"], re_formula="~1")
    result = model.fit()
     
     # Get p-value for the fixed effect of group
    p= result.pvalues.get("group[T.Gr2]", np.nan)
     
    if   tst==0: # on side test success Group1>  sucsess Group2
    
      if result.params["group[T.Gr2]"] < 0: # If the effect is positive (Group2 < Group1), 
            p= p / 2
      else:
            # If the effect is negative (Group2 < Group1), ignore for one-sided p-value calculation
            p= 1
    
    return(p)


def Generalized_estimating_equations(suc1,suc2,animals_sucess1,animals_sucess2,tst,num_trial):
    
      
     #create dataframe for LME
      n_total=len(animals_sucess1)+len(animals_sucess2)
      animal_id=[] 
      for animal in np.arange(n_total):
          animal_id=animal_id+[animal]*num_trial
          
      data = {
    'animal_id': animal_id,
    'group': ['GR1']*len(animals_sucess1)*num_trial + ['Gr2']*len(animals_sucess2)*num_trial,  
    'trial': list(range(1, num_trial+1))*n_total,  
    'success':list(suc1)+list(suc2) 
    }
      
      df = pd.DataFrame(data)
    
      model = smf.gee("success~ group", groups="animal_id", data=df, family=sm.families.Binomial())
      result = model.fit()

      coef = result.params["group[T.Gr2]"]
      p= result.pvalues["group[T.Gr2]"]
     

      if  tst==1: #two side test      
          std_err = result.bse["group[T.Gr2]"]
            
            # Calculate z-statistic
          z_stat = coef / std_err
            
            # Calculate two-sided p-value directly
          p = 2 * (1 - norm.cdf(abs(z_stat)))
      
      return(p)
      
      

test_methods = {
    'Ttest2': Ttest2,
    'Proportion2': Proportion2,
    'Fisher': Fisher,
    'Welch': Welch,
    'MannWhitney': MannWhitney,
    'Ttest1': Ttest1,
    'Proportion1': Proportion1,
    'Generalized_estimating_equations':Generalized_estimating_equations,
    'Multilevel_Mixed_Model':Multilevel_Mixed_Model
}

   
def run_simulation(
    mode='two-group',#  'two-group'
    method_name='Proportion2', # Ttest1 (compared to the chance level),Ttest2(compare 2 num_group),proportion1,proportion2,Wilcoxon1    nWhitney    ,Welch" ,'Permutation','MannWhitney'
    test_direction='Unilateral',

    num_trial=4,  # trial num_simulation for  each subject
    
    success_rate_grp1=60, # % average probability of succes of trained group
    std_grp1=20,# standard deviation of succes  between subjects in  trained group
    sample_size_grp1=6,

    chance_level_grp1=50., 

    success_rate_grp2=50, #% average probability of succes of control group  or chancelevel when only 1 experimental group
    std_grp2=20,  # standard deviation of succes  between subjects in control group
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
            * 'Ttest2', 'proportion2', 'Wilcoxon1', 'nWhitney', 'Welch' , 'Permutation','MannWhitney' 'Multilevel_Mixed_Model',Generalized_estimating_equations
    test_direction:str, 'Unilateral' |'Bilateral'
    num_trial : int, default 4
        Trial num_simulation for each subject
    success_rate_grp1 : float, default 70.
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
    num_simulation : int, default 1
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
        raise ValueError("mode must be 'one-group', 'two-group'")
    
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
        #'two-sided' for Type I error evaluation
        alternative = 1
    if  test_direction=='Unilateral':
        # 'greater' for power calculation
        alternative = 0
    if test_direction=='Bilateral':  alternative = 1 #two-sided for power calculation 

    rng = np.random.default_rng(seed=seed)
    
    print("\033[2mWORK IN PROGRESS")
    print()

    sample1=generate_sample(sample_size_grp1,prob1,std_grp1,tolerance, rng=rng)
    
    print ('This simulation was generated with the following sample parameters:')
    print()
    smpl1=sample1*100
    f_sample1 = ["{:.{}f}".format(val, 2) for val in smpl1]
    print('Groupe 1:')
    print('Probability of success of the diferent subjects:', f_sample1, '(%)')
    print (f"Average probability of  success:{np.mean(smpl1):.2f} (%)")
    print (f"Variability  of success :{np.std(smpl1):.2f} STD(%)")
    print()
    # if num_group==2:
    if mode == 'two-group':
        sample2=generate_sample(sample_size_grp2,prob2,std_grp2,tolerance, rng=rng)
        smpl2=sample2*100
        f_sample2 = ["{:.{}f}".format(val, 2) for val in smpl2]
        print('Group 2:')
        print('Probability of success of the diferent subjects:', f_sample2, '(%)')
        print (f"Average probability of  success:{np.mean(smpl2):.2f} (%)")
        print (f"Variability  of success :{np.std(smpl2):.2f} STD(%)")
        
        print()
    power=[]# ratio of false negatives
    for rep in range(num_simulation):
        fn=0#false negatives 
        fp=0#false positive when no difference betwen prob1 and prob2 (Type I error calculation)
        for it in range(num_iteration): 
            success1=[] #  succes cumulated over the animals
            animals_sucess1=[]# succes rate by animal
            for panimal in sample1:
                anim_suc=[]
                for trial in range(num_trial):
                    suc1 = rng.binomial(1, p=panimal, size=1)
                    success1.append(suc1[0])
                    anim_suc.append(suc1[0])
                animals_sucess1.append(np.sum(anim_suc)/num_trial)
            
            success1=np.array(success1)
            
           
            if mode == 'one-group':
                if method_name not in ['Ttest1','Proportion1']:
                    raise Exception ('Wrong test for compareson of one group with the chance level')
                pval=test_method_function(success1, animals_sucess1,prob2,alternative,num_trial) 
                
           
            elif mode == 'two-group':
                if method_name not in ['Ttest2','Proportion2','Fisher','Welch','MannWhitney','Generalized_estimating_equations','Multilevel_Mixed_Model']:
                    raise Exception ('Wrong test for compareson between  two num_group')
                
                success2=[]# all succes cumulated by group
                animals_sucess2=[]# succes rate by animal
                for panimal in sample2:
                    anim_suc=[]
                    for trial in range(num_trial):
                        suc2 = rng.binomial(1, p=panimal, size=1)
                        success2.append(suc2[0])
                        anim_suc.append(suc2[0])
                    animals_sucess2.append(np.sum(anim_suc)/num_trial)
                success2=np.array(success2)
              
                
                pval = test_method_function(success1,success2,animals_sucess1,animals_sucess2,alternative,num_trial)  

            if prob1==prob2 and pval<alpha_risk: #for Type I error calculation
                fp+=1

            if prob1>prob2 and pval>alpha_risk: # for power calculation
                fn+=1


        if prob1==prob2:
            tp1=f"{fp/num_iteration * 100:.2f}"
            print('Type I error simulation',rep+1, ' :',tp1,'%')
            power.append(fp/num_iteration*100)
        else :
            pwr=f"{100-fn/num_iteration*100:.2f}"
            print('Power simulation',rep+1, ' :',pwr,'%')
            power.append(100-fn/num_iteration*100)
    print()
    print("\033[2mDONE")
    if prob1==prob2:
        print()
        print(f"Type I error mean:{np.mean(power):.1f} %")
        if num_simulation>1:
            print(f"95% CI:{CI95(power):.1f}%")
    else:
        print()
        print(f"Mean Power:{np.mean(power):.1f} %")
        if num_simulation>1:
            print(f"95% CI:{CI95(power):.1f} %")


    
    
if __name__=="__main__":
    
        run_simulation(
            mode="two-group",
            method_name='Ttest2',
            test_direction='Bilateral',
           
            num_trial=4,             
            success_rate_grp1=70,
            std_grp1=10,
            sample_size_grp1=4,
    
    
    
            chance_level_grp1=50.,
    
            success_rate_grp2=50,
            std_grp2=20,
            
            sample_size_grp2=6,
    
            alpha_risk=0.05,
            num_iteration=5000,
            num_simulation=1,
            tolerance=0.05,
            seed=None,
           
        )
        