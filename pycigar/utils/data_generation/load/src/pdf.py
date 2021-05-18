import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import warnings

# Fit PDF
# Source: https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

# Create models from data

def make_pdf(dist, params, size=10000):
    """
    Generate distributions's Probability Distribution Function 
    """
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
    """
    For all time series load profiles, generate the corresponding means and standard deviations.
    """
    pdfs = []
    pdf_std = []

    for file in time_series:

        # mean and standard deviation
        params = st.norm.fit(file)
      
        pdf = make_pdf(st.norm, params)  
        
        pdf_std.append(params[1])
        pdfs.append(pdf)
        
    return pdfs, pdf_std
