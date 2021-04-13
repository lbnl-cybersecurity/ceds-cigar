import pandas as pd
import numpy as np 

from pycigar.utils.data_generation.load.src.pdf import normalize_data
from pycigar.utils.data_generation.load.src.cdf import make_cdf, compare_cdfs
from pycigar.utils.data_generation.load.src.stochastic_computations import generate_mean_reversion_rate, generate_polynomial, euler_maruyama_method, euler_maruyama_method_old
from pycigar.utils.data_generation.load.src.autocorrelation import autocorrelation, summed_autocorrelation


def generate_stochastic_load(f, ts, input_time, output_time, order, fn):
    # Called by user
    mrr, em_mu, em_x, cdfs_data = resample_meaned_data(f, ts, input_time, output_time, order, fn)
    return mrr, em_mu, em_x, cdfs_data


def resample_meaned_data(file, time_series, input_time, output_time, order, fn):
    # User never calls
    """
    
    Args: 
    input_file (list of strings) - CSV file names 
        ex. ['7_MWp_P.csv', '10_MWp_P.csv', '12_MWp_P.csv', '19_MWp_P.csv']
    input_time (string) - Input time sequence 
        ex. '1S'
    output_time (string)- Output time sequence 
        ex. '1S'
    order (list of integers) -  MWp for each file 
        ex. [7, 10, 12, 19]
    """

    #############
    print('normalize data\n\n')
    pdfs, std_of_nonnormal_pdfs = normalize_data(time_series, order)

    #############
    print('generate polynomial \n \n ')
    spl = generate_polynomial(pdfs, order, std_of_nonnormal_pdfs)

    #############
    print('cdfs')
    cdfs_data = make_cdf(time_series, order, "Measured")
    meaned_data = []
    for k in range(len(file)):
        meaned_data.append(file[k].resample(output_time, label='right').mean())

    #############
    print('math\n \n')
    mrr, em_mu, em_x, index_vals, meaned_data = generate_mean_reversion_rate(
        file, output_time, input_time, order, spl)
    cdfs_real, all_autocor = euler_maruyama_method_old(
        em_mu, em_x, order, mrr, output_time, index_vals, spl)

    #############
    print('autocorrelation')
    autocorrelation(meaned_data, all_autocor, output_time)
    summed_autocorrelation(cdfs_data, cdfs_real, output_time)

    #############
    print('compare cdfs')
    compare_cdfs(cdfs_real, cdfs_data, order)

    em_mu_df = pd.DataFrame(np.array(em_mu)[:, :, 0]).transpose()
    em_x_df = pd.DataFrame(np.array(em_x)[:, :, 0]).transpose()
    em_mu_df.columns = fn
    em_x_df.columns = fn

    return mrr, em_mu_df, em_x_df, cdfs_data


class LoadGenerator:
    def __init__(self, data, IEEE_load, input_time='15T', output_time='1S'):
        #order = self.order_file = [int(d.split('/')[-1].split('_')[0]) for d in data]
        order =[7, 10, 12, 19]

        self.input_time = input_time
        self.output_time = output_time
        file = [] # series from csv file
        file_diff = [] # resampled data
        time_series = [] # resampled data as a series

        max_IEEE_load = max(IEEE_load) * 10**3  
        self.scale_factor = max_IEEE_load / (19 * 10**3) 
        self.adjusted_order = np.zeros((len(IEEE_load)))
        
        for load in range(len(IEEE_load)):
            # take the IEEE loads and convert them to be of similar scale to the 7, 10, 12, and 19 MW 
            # produces a new "order" array
            self.adjusted_order[load] = IEEE_load[load] / self.scale_factor

        for i, f in enumerate(data):
            file.append(pd.read_csv(f, index_col = 0, header = None,names=['P'], parse_dates=True, infer_datetime_format=True))
            file_diff.append(file[i].resample(output_time).mean().diff(1).dropna())
            time_series.append(file_diff[i].iloc[:,0])

        def closest(lst, K):      
            return lst.index(lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))])

        self.file_idx = np.zeros(len(self.adjusted_order))
        adjusted_time_series = []
        adjusted_file = []

        for load in range(len(self.adjusted_order)):
            close_idx = closest(order, self.adjusted_order[load])
            self.file_idx[load] = close_idx #pick the file that is closest to the scaled dss load
            adjusted_time_series.append(time_series[close_idx] * self.scale_factor) #scale the nearest time series data 
            adjusted_file.append(file[close_idx] * self.scale_factor) #scale the nearest file
   
        pdfs, std_of_nonnormal_pdfs = normalize_data(adjusted_time_series, self.adjusted_order)#time_series, order)
        self.spl = generate_polynomial(pdfs, self.adjusted_order, std_of_nonnormal_pdfs)
   
        meaned_data = []
        for i, f in enumerate(adjusted_file):
            meaned_data.append(f.resample(output_time, label='right').mean())
        #self.mrr, self.em_mu, self.em_x, self.index_vals, _ = generate_mean_reversion_rate(file, output_time, input_time, order, self.spl)
        
        self.mrr, self.em_mu, self.em_x, self.index_vals, _ = \
                generate_mean_reversion_rate(adjusted_file, output_time, input_time, self.adjusted_order, self.spl)

    def generate_load(self, order):
        all_autocor = euler_maruyama_method(self.em_mu, self.em_x, self.adjusted_order, self.mrr, self.output_time, self.index_vals, self.spl, self.adjusted_order)
         
        for i in range(len(all_autocor)):
            all_autocor[i] = all_autocor[i] / self.scale_factor


        return all_autocor