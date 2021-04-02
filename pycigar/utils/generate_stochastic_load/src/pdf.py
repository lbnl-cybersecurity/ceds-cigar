import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import warnings

# Fit PDF
# Source: https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

# Create models from data


def best_fit_distribution(data, bins=200, ax=None):
    """
    Fit data to normal distribution
    """
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    params = st.norm.fit(data)

    # # Separate parts of parameters
    # arg = params[:-2]
    # loc = params[-2]
    # scale = params[-1]

    # # Calculate fitted PDF and error with fit in distribution
    # pdf = st.norm.pdf(x, loc=loc, scale=scale, *arg)

    return (st.norm.name, params)


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get same start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc,
                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc,
                   scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def normalize_data(time_series, order):
    i = 0
    pdfs = []
    std_of_nonnormal_pdfs = []

    for file in time_series:
        # Find best fit distribution
        best_fit_name, best_fit_params = best_fit_distribution(file, 200, None)
        best_dist = getattr(st, best_fit_name)

        # Make PDF with best params
        pdf = make_pdf(best_dist, best_fit_params)
        pdfs.append(pdf)

        _, s = st.norm.fit(file)  # mean and standard deviation
        std_of_nonnormal_pdfs.append(s)

        # Plot PDF
        # Display
        plt.figure(figsize=(5, 2))
        ax = pdf.plot(lw=2, label='PDF', legend=True)
        file.plot(kind='hist', bins=25, density=True,
                  alpha=0.5, label='Data', legend=True, ax=ax)

        # Set the parameters
        param_names = (best_dist.shapes + ', loc, scale').split(
            ', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v)
                               for k, v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)

        # Label the graph
        ax.set_title(
            u'Normalized ' + str(order[i]) + ' MW Probability Distribution Function\n' + dist_str)
        ax.set_xlabel(u'P(t) - P(t-1)')
        ax.set_ylabel('Frequency')

        i += 1

    return pdfs, std_of_nonnormal_pdfs
