import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from src.cdf import make_cdf

# Stochastic Load Computations


def generate_mean_reversion_rate(files, output_time, input_time, order, spl):
    """
    Finds the mean reversion rate

    Args: 
    files (list of DataFrames)
    input_time (string) - represents in the input time series, ex. '1S'
    output_time (string) - represents the output time series, ex. '1S'
    order (list) - loading levels
    """

    # x(t) profile
    mrr = []
    em_mu = []
    em_x = []
    em_Y = []
    index_vals = []
    meaned_data = []
    pd_read_data = []

    for k in range(len(files)):
        file1_mean = files[k].resample(output_time, label='right').mean()
        meaned_data.append(file1_mean)
        pd_read_data.append(files[k])

    for n in range(len(meaned_data)):
        # mu(t) profile
        mu10 = pd_read_data[n].resample(input_time).mean()
        mu10 = mu10.resample(output_time).pad()
        em_mu.append(mu10)

        # x profile
        index_difference = len(file1_mean) - len(mu10)
        x_10 = meaned_data[n].iloc[:len(meaned_data[n])-index_difference, :]
        index_vals.append(x_10.index.array)
        em_x.append(x_10)

        # Forming the Y's (contingent on std_dev)
        Y = np.array([])
        np_x_10 = x_10.to_numpy()
        np_mu_10 = mu10.to_numpy()

        std_dev = splev(order[n], spl)

        for i in range(len(x_10)):
            Y = np.append(Y, (np_mu_10[i][0] - np_x_10[i][0]) / (std_dev**2))

        em_Y.append(Y)

        # Mean reversion rate
        num = 0
        denom = 0
        for j in range(len(x_10)-1):
            num += Y[j] * (np_x_10[j+1][0] - np_mu_10[j+1][0])
            denom += Y[j] * (np_x_10[j][0] - np_mu_10[j][0])
        mean_rev_rate = -np.log(num/denom)
        mrr.append(mean_rev_rate)

    for i in range(len(em_mu)):
        em_mu[i] = em_mu[i].to_numpy()
        em_x[i] = em_x[i].to_numpy()

    #     fig = plt.figure(figsize=(12, 8))
    #     plt.plot(em_x[i], label='x data')
    #     plt.plot(em_mu[i], label='mu data')
    #     plt.xlabel('Time')
    #     plt.ylabel('Demand [W]')
    #     plt.title('Demand throughout a portion of the day for ' +
    #               str(order[i]) + ' MWp')
    #     plt.legend()
    #     plt.show()

    return mrr, em_mu, em_x, index_vals, meaned_data
    # return mrr, em_mu, em_x, meaned_data


# Source: https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method

def euler_maruyama_method(em_mu, em_x, order, mrr, output_time, index_values,  spl):
    """
    Runs the Euler Maruyama method on different loading levels

    Args: 
    em_mu (list of arrays) - array of the mean, zero-order hold , each index is at a different loading level
    em_x (list of arrays) - array of the demand data, condensed based on upsampling, each index is at a different loading level
    p (function) - polynomial function representing standard deviation
    mrr (array of integers) - contains the mean reversion rate for the different loading levels
    output_time (string) - represents the output time series, ex. '1S'
    index_values (list of arrays) - represents the time related to each data point, each index is at a different loading level
    order (list) - loading levels
    """
    all_autoc = []

    for i in range(len(order)):
        k = np.random.randint(0, len(em_mu))
        t_init = 0
        t_end = len(em_x[k]) - 1
        N = len(em_x[k]) - 1
        dt = 1
        x_init = em_x[k][0]

        c_theta = float(mrr[k])
        c_sigma = splev(order[i], spl)

        def mu_funct(y, t, c_mu):
            """Implement the Ornstein–Uhlenbeck mu."""  # = \theta (\mu-Y_t)
            return c_theta * (c_mu - y)

        def sigma(y, t):
            """Implement the Ornstein–Uhlenbeck sigma."""  # = \sigma
            return c_sigma

        def dW(delta_t):
            """Sample a random number at each call."""
            return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

        ts = np.arange(t_init, t_end, dt)
        xs = np.zeros(N)

        xs[0] = x_init
        # plt.figure(figsize=(10, 7))
        # plt.xlabel("Time step (seconds)")
        # plt.ylabel("Active Power [kW]")
        # plt.title("Performing " + str(num_sims) + " simulations using the Euler Maruyama Method on the "
        #           + str(order[k])+" MW data ")

        for i in range(1, ts.size):
            t = (i-1) * dt
            x = xs[i-1]
            xs[i] = x + mu_funct(x, t, em_mu[k][i]) * \
                dt + sigma(x, t) * dW(dt)

            # plt.plot(ts, xs/1e3, linewidth=1)

        # plt.plot(em_mu[k] / 1e3, lw=3, color='y')
        # plt.legend(('Stochastic Load Profile 1', 'Stochastic Load Profile 2',
        #             'Stochastic Load Profile 3', 'Expected 15 minute demand'))
        # plt.show()

        #all_xs_diff.append(pd.Series(xs).diff(1).dropna())

        #write_EM2_csv(pd.Series(xs).dropna(),
        #              index_values[k], output_time, order[k], N)
        #write_EM2_csv(pd.Series(xs).dropna(), output_time, order[k], N)
        eM_autocor = pd.Series(xs)
        eM_autocor.index = index_values[k][:N]
        all_autoc.append(eM_autocor)

    #cdfs_real = make_cdf(all_xs_diff, order, "Realized")
    return all_autoc


def write_EM2_csv(xs, iv, output_time, loading_level, N):
    """
    Writes the output Series of the Euler Maruyama function to a CSV document

    Args: 
    xs (Series) - Euler Maruyama realized data
    file_diff (array of datetime objects) - time stamps, each index represents a different loading level
    output_time (string) - such as '1S'
    loading_level - integer such as 7 for 7 MWp
    N (integer) - number of trials run in Euler Maruyama method

    """
    xs.index = iv[:N]
    xs.to_csv("data/"+str(loading_level) + "_MWp_" + output_time + ".csv")


def generate_polynomial(pdfs, order, std_of_nonnormal_pdfs):
    """
    Fits a curve to standard deviation of the load profiles
    Saves this curve to a pickle file
    Plots curve
    """

    spl = splrep(order, std_of_nonnormal_pdfs)


    # pdfs_std = []
    # for i in range(len(pdfs)):
    #     pdfs_std.append(np.std(pdfs[i].index.tolist()))

    # plt.figure()
    # x2 = np.linspace(order[0], order[len(order)-1], 200)
    # y2 = splev(x2, spl)
    # y2 = np.true_divide(y2, 1000)
    # pdfs_std = np.true_divide(std_of_nonnormal_pdfs, 1000)
    # plt.plot(order, pdfs_std, 'o', x2, y2)
    # plt.title('σ curve given different loading levels')
    # plt.xlabel('Loading Level [MWp]')
    # plt.ylabel('σ [kW]')
    # plt.show()

    return spl
