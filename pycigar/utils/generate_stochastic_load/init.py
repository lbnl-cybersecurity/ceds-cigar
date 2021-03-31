from src.pdf import normalize_data
from src.cdf import make_cdf, compare_cdfs
from src.stochastic_computations import generate_mean_reversion_rate, generate_polynomial, euler_maruyama_method
from src.autocorrelation import autocorrelation, summed_autocorrelation


def generate_stochastic_load(f, ts, input_time, output_time, order):
    # Called by user
    for j in range(len(output_time)):
        mrr, em_mu, em_x, cdfs_data = resample_meaned_data(
            f, ts, input_time, output_time[j], order)
    return mrr, em_mu, em_x, cdfs_data


def resample_meaned_data(file, time_series, input_time, output_time, order):
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
    cdfs_real, all_autocor = euler_maruyama_method(
        em_mu, em_x, order, mrr, output_time, index_vals, spl)

    #############
    print('autocorrelation')
    autocorrelation(meaned_data, all_autocor, output_time)
    summed_autocorrelation(cdfs_data, cdfs_real, output_time)

    #############
    print('compare cdfs')
    compare_cdfs(cdfs_real, cdfs_data, order)

    return mrr, em_mu, em_x, cdfs_data
