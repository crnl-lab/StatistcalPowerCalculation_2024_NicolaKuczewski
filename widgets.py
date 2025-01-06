import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


import ipywidgets as W

from statistical_power_calculation import run_simulation, test_methods

def show_widget():
    # make widgets
    mode = W.Dropdown(value='one-group', options=['one-group', 'two-group'], ensure_option=True, description="Mode",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    method_name = W.Dropdown(value='Ttest1', options=list(test_methods.keys()), ensure_option=True, description="Test method",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    test_direction=W.Dropdown(value='Unilateral', options=['Unilateral','Bilateral'],ensure_option=True, description="Mode",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    num_trial = W.IntText(value=4, description="Number of trials",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    
    success_rate_grp1 = W.FloatText(value=70., description="Success rate Gr1", style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    std_grp1 = W.FloatText(value=10., description="Variability Gr1",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    sample_size_grp1 = W.IntText(value=4, description="Sample size Gr1",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    
    chance_level_grp1= W.FloatText(value=50., description="Chance level Gr1",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    
    success_rate_grp2 = W.FloatText(value=30., description="Success rate Gr2",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    std_grp2 = W.FloatText(value=25., description="Variability Gr2",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    sample_size_grp2 = W.IntText(value=6, description="Sample size Gr2",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    
    alpha_risk = W.FloatText(value=0.05, description="Alpha risk",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    num_iteration = W.IntText(value=5000, description="Number of iterations",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    num_simulation = W.IntText(value=1, description="Repetition",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    tolerance = W.FloatText(value=0.05, description="Tolerance error in generated sample mean and std",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    button = W.Button(description="run simulation",style={'description_width': 'initial'},layout=W.Layout(width='400px'))
    out = W.Output(layout={'border': '1px solid black'})
    
    hbox = W.VBox(children=[mode, method_name,test_direction, num_trial, 
                            success_rate_grp1, std_grp1, sample_size_grp1,
                            chance_level_grp1,
                            success_rate_grp2,std_grp2, sample_size_grp2,
                            alpha_risk,
                            num_iteration,num_simulation,tolerance,
                            button, out])
    
    @out.capture()
    def on_mode_changed(change=None):
        
        if mode.value == 'one-group':
            chance_level_grp1.disabled = False
            success_rate_grp2.disabled = True
            std_grp2.disabled = True
            sample_size_grp2.disabled = True
            method_name.options = ['Ttest1','Proportion1']
            test_direction.options=['Unilateral']
        
        elif mode.value == 'two-group':
            chance_level_grp1.disabled = True
            success_rate_grp2.disabled = False
            std_grp2.disabled = False
            sample_size_grp2.disabled = False
            method_name.options = ['Ttest2','Proportion2','Fisher','Welch','MannWhitney','Generalized_estimating_equations','Multilevel_Mixed_Model']
            test_direction.options=['Unilateral','Bilateral']
    
    
    mode.observe(on_mode_changed)
    on_mode_changed()
    
    @out.capture()
    def on_button_run_clicked(change=None):
        out.clear_output()
        run_simulation(
            mode=mode.value,
            method_name=method_name.value,
            test_direction=test_direction.value,
            
            num_trial=num_trial.value,
    
            success_rate_grp1=success_rate_grp1.value,
            std_grp1=std_grp1.value,
            sample_size_grp1=sample_size_grp1.value,
    
            chance_level_grp1=chance_level_grp1.value,
    
            success_rate_grp2=success_rate_grp2.value,
            std_grp2=std_grp2.value,
            sample_size_grp2=sample_size_grp2.value,
            
            alpha_risk=alpha_risk.value,
            num_iteration=num_iteration.value,
            num_simulation=num_simulation.value,
            tolerance=tolerance.value,
        )
    button.on_click(on_button_run_clicked)
    
    display(hbox)
    