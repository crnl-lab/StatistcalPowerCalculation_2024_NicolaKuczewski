"""
Created on 14/06/2018

@author: Nicola Kuczewski and Samuel Garcia
"""
from scipy.stats import beta
from scipy.optimize import minimize
import scipy.stats
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import  norm
import warnings
warnings.filterwarnings("ignore")


def fit_beta_parameters(target_mean, target_std, min_val, max_val):
    """
    Compute alpha and beta for a scaled Beta distribution that approximates
    the target mean and std over [min_val, max_val].
    """
    scaled_mean = (target_mean - min_val) / (max_val - min_val)
    scaled_std = target_std / (max_val - min_val)
    

    def objective(params):
        a, b = params
        if a <= 0 or b <= 0:
            return np.inf
        mean = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1))
        return (mean - scaled_mean) ** 2 + (np.sqrt(var) - scaled_std) ** 2

    result = minimize(objective, [2, 2], bounds=[(1e-3, None), (1e-3, None)])
    return result.x  # alpha, beta

def sample_generation(alpha, beta_param, min_val, max_val, n):
    """
    Generate a sample from a Beta(alpha, beta) scaled to [min_val, max_val]
    """
    samples = beta.rvs(alpha, beta_param, size=n)
    scaled_samples = min_val + samples * (max_val - min_val)
    return scaled_samples


def CI95(data):
    CI=scipy.stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=scipy.stats.sem(data)) 
    
    return np.mean(data)-CI[0]
   
    
def Welch (suc1,suc2,animals_sucess1,animals_sucess2,tst,num_trial):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided’'
    p = scipy.stats.ttest_ind(animals_sucess1,animals_sucess2,equal_var=False,alternative=alter)[1]
    return p


    
def Ttest1(success1, animals_sucess1,chance,tst,num_trial):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    p=scipy.stats.ttest_1samp(animals_sucess1,chance/100,alternative=alter)[1]
    return p

def Ttest2(success1,success2,animals_sucess1,animals_sucess2,tst,num_trial):
    if tst==0:
        alter='greater'
    else:
        alter='two-sided'
    p=scipy.stats.ttest_ind(animals_sucess1,animals_sucess2,alternative=alter)[1]
    return p
    

def Proportion1(success1, animals_sucess1,chance,tst,num_trial):
  
    if tst==0:
      alter='larger'
    else:
      alter='two-sided'
    succes=np.sum(success1)
    total= len(success1)
    p=proportions_ztest(succes,total,value=chance/100,alternative=alter)[1]
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
    return p
               

    

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
    
    return p

def Multilevel_Mixed_Model1(suc1, animals_success1, chance,tst,num_trial):
    chance=chance/100
    n_animals = len(animals_success1)
    
    # Real group
    animal_id = list(np.repeat(np.arange(n_animals), num_trial))
    group = ['real'] * len(suc1)
    success = list(suc1)

    # Synthetic group (virtual animals with fixed success rate)
    animal_id += list(np.repeat(np.arange(n_animals, 2*n_animals), num_trial))
    group += ['theoretical'] * (n_animals * num_trial)
    success += [chance] * (n_animals * num_trial)

    # Build DataFrame
    df = pd.DataFrame({
        'animal_id': animal_id,
        'group': group,
        'success': success
    })

    df['group'] = df['group'].astype('category')

    # Fit mixed model: random effect on animal_id
    model = smf.mixedlm("success ~ group", df, groups=df["animal_id"], re_formula="~1")
    result = model.fit()

    # Extract p-value
    p = result.pvalues.get("group[T.theoretical]", np.nan)

    # One-sided test: real group > theoretical
    if tst == 0:
        if result.params["group[T.theoretical]"] < 0:  # theoretical < real → good
            p = p / 2
        else:
            p = 1
    
    
    return p

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
      
def Generalized_estimating_equations1(suc1, animals_success1,chance_level, tst, num_trial):
    # Create DataFrame
    n_total = len(animals_success1)
    animal_id = []
    
    for animal in range(n_total):
        animal_id += [animal] * num_trial

    data = {
        'animal_id': animal_id,
        'trial': list(range(1, num_trial+1)) * n_total,
        'success': list(suc1)
    }

    df = pd.DataFrame(data)
   

    # Add offset: logit of the chance level
    logit_chance = np.log(chance_level/100 / (1 - chance_level/100))
    df["offset"] = logit_chance
 
    # Fit GEE model with offset
    model = smf.gee("success ~ 1", groups="animal_id", data=df, 
                    family=sm.families.Binomial(), offset=df["offset"])
    result = model.fit()

    coef = result.params["Intercept"]
    p = result.pvalues["Intercept"]

    # Two-sided test (override p if tst == 1)
    if tst == 1:
        std_err = result.bse["Intercept"]
        z_stat = coef / std_err
        p = 2 * (1 - norm.cdf(abs(z_stat)))

    return p      

test_methods = {
    'Ttest2': Ttest2,
    'Proportion2': Proportion2,
    'Welch': Welch,
    'Ttest1': Ttest1,
    'Proportion1': Proportion1,
    'Generalized_estimating_equations':Generalized_estimating_equations,
    'Generalized_estimating_equations1':Generalized_estimating_equations1, # for comparison with chance level
    'Multilevel_Mixed_Model':Multilevel_Mixed_Model,
    'Multilevel_Mixed_Model1':Multilevel_Mixed_Model1, # for comparison with chance level
}

   
def run_simulation(
    mode='two-group',#  'two-group'
    method_name='Proportion2', # Ttest1 (compared to the chance level),Ttest2(compare 2 num_group),proportion1,proportion2,Wilcoxon1    nWhitney    ,Welch" ,'Permutation','MannWhitney'
    test_direction='Unilateral',

    num_trial=8,  # trial num_simulation for  each subject
    
    success_rate_grp1=70, # % average probability of succes of trained group
    std_grp1=20,# standard deviation of succes  between subjects in  trained group
    sample_size_grp1=4,
    

    chance_level=50., 

    success_rate_grp2=50, #% average probability of succes of control group  or chancelevel when only 1 experimental group
    std_grp2=0,  # standard deviation of succes  between subjects in control group
    sample_size_grp2=6,


    alpha_risk=0.05,    
    num_iteration=5000,
    num_simulation=5, # number or simulations repetition for calculate the average power and 95%CI
    seed=None,
    maximum_success_rate=100,
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
    seed: Nont or int
        The generation seed
    """
    
    if method_name not in test_methods:
        raise ValueError(f"method_name={method_name} must be in {list(test_methods.keys())}")
    test_method_function = test_methods[method_name]

    if mode not in ('one-group', 'two-group'):
        raise ValueError("mode must be 'one-group', 'two-group'")
    if mode in('two-group'):
        
        if (method_name=='Proportion2'and std_grp2!=0) : 
            print()
            print(" WARNING: THE USE OF PROPORTION TEST INFLATE TYPE I ERROR WHEN THE VARIABILITY OF GROUP2 IS NOT ZERO")
            print()
    
        if success_rate_grp1<success_rate_grp2 :
            raise ValueError("Sucecs rate of group 1 should be ≥ of success rate of groupe 2")
       
        if success_rate_grp2<chance_level:raise ValueError(" The success rate of Group 2 cannot be lower than chance level") 
    
    if success_rate_grp1==success_rate_grp2 or success_rate_grp1==chance_level: 
        #'two-sided' for Type I error evaluation
        alternative = 1
    else:
        if  test_direction=='Unilateral':
            # 'greater' for power calculation
            alternative = 0
        if test_direction=='Bilateral':  alternative = 1 #two-sided for power calculation 

    minimum_success_rate=chance_level
    alpha1, beta1 = fit_beta_parameters(success_rate_grp1,std_grp1, minimum_success_rate, maximum_success_rate)
    alpha2, beta2 = fit_beta_parameters(success_rate_grp2,std_grp2, minimum_success_rate, maximum_success_rate)
    rng = np.random.default_rng(seed=seed)
   
    
    print("\033[2mWORK IN PROGRESS")
    power=[]# ratio of false negatives

    
    for rep in range(num_simulation):
        print()
        print ('simulation:'.upper(),rep+1)
        tp=0#true positif, for power calculation
        fp=0#false positive when no difference betwen success_rate_grp1 and success_rate_grp2 (Type I error calculation)
        stds1=[]
        stds2=[]
        means1_s=[]
        means2_s=[]
        for it in range(num_iteration): 
            # generation sample 1
            if std_grp1==0: sample1=[success_rate_grp1] *sample_size_grp1
            else:
                sample1=sample_generation (alpha1, beta1, minimum_success_rate, maximum_success_rate,sample_size_grp1)
            stds1.append(np.std(sample1))
            means1_s.append(np.mean(sample1))
            
            success1=[] #  succes cumulated over the animals
            animals_sucess1=[]# succes rate by animal 
            for panimal in sample1:
                anim_suc=[]
                for trial in range(num_trial):
                    suc1 = rng.binomial(1, p=panimal/100, size=1)
                    success1.append(suc1[0])
                    anim_suc.append(suc1[0])
                animals_sucess1.append(np.sum(anim_suc)/num_trial)
           
            success1=np.array(success1)
            
           
            if mode == 'one-group':
                if method_name not in ['Ttest1','Proportion1','Multilevel_Mixed_Model1','Generalized_estimating_equations1']:
                    raise Exception ('Wrong test for compareson of one group with the chance level')
                pval=test_method_function(success1, animals_sucess1,chance_level,alternative,num_trial) 
                
                
           
            elif mode == 'two-group':
                if method_name not in ['Ttest2','Proportion2','Fisher','Welch','MannWhitney','Generalized_estimating_equations','Multilevel_Mixed_Model']:
                    raise Exception ('Wrong test for compareson between  two num_group')
                
                success2=[]# all succes cumulated by group
                animals_sucess2=[]# succes rate by animal
                # generation sample 2
                if std_grp2==0: sample2=[success_rate_grp2] *sample_size_grp2
                else:
                    sample2=sample_generation (alpha2, beta2, minimum_success_rate, maximum_success_rate,sample_size_grp2)
                stds2.append(np.std(sample2))
                means2_s.append(np.mean(sample2))
               
                for panimal in sample2:
                    anim_suc=[]
                    for trial in range(num_trial):
                        suc2 = rng.binomial(1, p=panimal/100, size=1)
                        success2.append(suc2[0])
                        anim_suc.append(suc2[0])
                    animals_sucess2.append(np.sum(anim_suc)/num_trial)
                success2=np.array(success2)
              
                
                pval = test_method_function(success1,success2,animals_sucess1,animals_sucess2,alternative,num_trial)  

            if success_rate_grp1==success_rate_grp2 and pval<alpha_risk: # False positif for Type I error calculation
                fp+=1

            if success_rate_grp1>success_rate_grp2 and pval<alpha_risk: # True positif for power calculation
                tp+=1

        print ( ' Estimated means groupe 1: ',np.mean(means1_s))
        print ( ' Estimated std groupe 1:',np.mean(stds1))
        print ( ' Estimated means groupe 2: ',np.mean(means2_s))
        print ( ' Estimated std groupe 2:',np.mean(stds2))
        print()
        if success_rate_grp1==success_rate_grp2:
            tp1=f"{fp/num_iteration * 100:.2f}"
            print('Type I error simulation',rep+1, ' :',tp1,'%')
            power.append(fp/num_iteration*100)
        else :
            pwr=f"{tp/num_iteration*100:.2f}"
            print('Power simulation',rep+1, ' :',pwr,'%')
            power.append(tp/num_iteration*100)
    print()
    print("\033[2mDONE")
    if success_rate_grp1==success_rate_grp2:
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
            method_name='Proportion2',
            test_direction='Unilateral',
            num_trial=8,
            chance_level=1/2*100.,
            alpha_risk=0.05,
            num_iteration=5000,
            num_simulation=4,
            seed=None,
            # parameter Group 1
            success_rate_grp1=70,
            std_grp1=20,
            sample_size_grp1=4,
            
            # parameter Group 2
            success_rate_grp2=50,
            std_grp2=0,
            sample_size_grp2=4,
            
            
           
        )
        