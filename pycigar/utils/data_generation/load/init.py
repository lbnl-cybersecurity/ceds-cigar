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
    print('math junk \n \n')
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
    def __init__(self, data, input_time='15T', output_time='1S'):
        order = self.order_file = [int(d.split('/')[-1].split('_')[0]) for d in data]
        self.input_time = input_time
        self.output_time = output_time
        file = [] # series from csv file
        file_diff = [] # resampled data
        time_series = [] # resampled data as a series

        for i, f in enumerate(data):
            file.append(pd.read_csv(f, index_col = 0, header = None,names=['P'], parse_dates=True, infer_datetime_format=True))
            file_diff.append(file[i].resample(output_time).mean().diff(1).dropna())
            time_series.append(file_diff[i].iloc[:,0])

        # calculate em_mu...
        pdfs, std_of_nonnormal_pdfs = normalize_data(time_series, order)
        self.spl = generate_polynomial(pdfs, order, std_of_nonnormal_pdfs)
        meaned_data = []
        for i, f in enumerate(file):
            meaned_data.append(f.resample(output_time, label='right').mean())
        self.mrr, self.em_mu, self.em_x, self.index_vals, _ = generate_mean_reversion_rate(file, output_time, input_time, order, self.spl)

    def generate_load(self, order):
        all_autocor = euler_maruyama_method(self.em_mu, self.em_x, order, self.mrr, self.output_time, self.index_vals, self.spl, self.order_file)
        return all_autocor