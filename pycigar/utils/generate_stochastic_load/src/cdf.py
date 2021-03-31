import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def make_cdf(time_series, order, title_tag):
    """
    Generates and returns the cumulative distribution function of a time-series

    Args: 
    time_series (list of Series) - time series data of measured load profile
    order (list) - loading levels
    title_tag (string) - describes time series (ex. 'measured', 'generated')
    """
    cdfs = []
    for n in range(len(time_series)):
        cdf_temp = time_series[n].sort_values()/1e3
        cum_dist = np.linspace(0., 1., len(cdf_temp))
        off_cdf = pd.Series(cum_dist, index=cdf_temp)
        plt.figure()
        off_cdf.plot()

        plt.xlabel('P(t)-P(t-1) [kw]')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of ' + title_tag +
                  " Data for Loading Level of " + str(order[n]) + " MWp")
        cdfs.append(off_cdf)
    return cdfs


def compare_cdfs(cdfs_real, cdfs_data, order):
    """
    Plots the demand and realized data's CDFs on one figure

    Args: 
    cdfs_real (list of Series) - demand data CDF, each index represents a different loading level
    cdfs_data (list of Series) - realized data CDF, each index represents a different loading level
    order (list) - loading levels
    """

    for i in range(len(cdfs_real)):
        plt.figure()
        plt.plot(cdfs_real[i], 'b-', label="Realization data")
        plt.plot(cdfs_data[i], 'r+', label="Measured data")
        plt.title("CDF at loading level of " + str(order[i]) + " MWp")
        plt.legend()

        plt.show()
