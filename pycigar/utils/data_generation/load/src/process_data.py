import pandas as pd

def read_initial_data(input_file, input_time, output_time, order):
    # Called by user
    """ 
    Returns contents of input CSV files as different output time sequences

    Args: 
    input_file (list of strings) - CSV file names 
        ex. ['7_MWp_P.csv', '10_MWp_P.csv', '12_MWp_P.csv', '19_MWp_P.csv']
    input_time (string) - Input time sequence 
        ex. '1S'
    output_time (list of strings)- Output time sequences 
        ex. ['1S', '10S', '30S']
    order (list of integers) -  MWp for each file 
        ex. [7, 10, 12, 19]

    """
    ff = []
    fdd = []
    tss = []
    for n in range(len(output_time)):
        f, fd, ts = read_files(input_file, input_time, output_time[n])
        ff.append(f)
        fdd.append(fd)
        tss.append(ts)
    return ff[0], fdd[0], tss[0]


def read_files(input_file, input_time, output_time):
    # User does not call

    """ 
    Reads the input CSV files and resamples/diffs
    Returns data in different forms

    Args: 
    input_file (list of strings) - CSV file names 
        ex. ['7_MWp_P.csv', '10_MWp_P.csv', '12_MWp_P.csv', '19_MWp_P.csv']
    input_time (string) - Input time sequence 
        ex. '1S'
    output_time (list of strings)- Output time sequences 
        ex. ['1S', '10S', '30S']

    """
    file = [] # series from csv file
    file_diff = [] # resampled data
    time_series = [] # resampled data as a series

    for n in range(len(input_file)):
        file.append(pd.read_csv(input_file[n], index_col = 0, header = None,names=['P'], parse_dates=True ,infer_datetime_format=True))
        file_diff.append(file[n].resample(output_time).mean().diff(1).dropna())
        time_series.append(file_diff[n].iloc[:,0])

    return file, file_diff, time_series