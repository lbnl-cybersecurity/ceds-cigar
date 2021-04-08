import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

def autocorrelation(meaned_data, all_autocor, output_time):
    '''
    Shows autocorrelation of measured data matches
    the autocorrelation of a stochastically generated load profile
    '''

    order = [7, 10, 12, 19]

    for i in range(len(meaned_data)):
        ax = plt.gca()
    
        autocorrelation_plot(meaned_data[i], label='Measured Data', ax = ax)
        autocorrelation_plot(all_autocor[i].resample(output_time, label='right').mean(), label='Stochastic Load Profile', ax = ax)
        plt.title('Autocorrelation between Measured and Stochastically Generated Profile for ' + str(order[i]) + ' MWp')
        
        plt.legend()
        plt.show()   

def summed_autocorrelation(meaned_data, all_autocor, output_time):
    '''
    Shows autocorrelation of 7 stochastically generated 1 MW load profiles matches
    the autocorrelation of a stochastically generated 7 MW load profile
    '''
    order = [7, 10, 12, 19]

    for i in range(len(meaned_data)):
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        autocorrelation_plot(meaned_data[i], label='Sum of independent stochastic load profiles', ax = ax)
        autocorrelation_plot(all_autocor[i], label='7MW stochastic load profile', ax = ax)
        plt.title('Autocorrelation between sum of independent stochastic load profiles and Stochastically Generated Profile for ' + str(order[i]) + ' MWp')
       
        plt.legend()
        plt.show()   