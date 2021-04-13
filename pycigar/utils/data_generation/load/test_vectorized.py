# from init import *
# from src.process_data import read_initial_data

# data = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/utils/data_generation/load/data'
# fi, fj, fk = read_initial_data([data + '/7_MWp_P.csv', data + '/10_MWp_P.csv', data + '/12_MWp_P.csv', data + '/19_MWp_P.csv'], '15T', ['1S' ], [7, 10, 12, 19])

# fn = ['7_MWp_P.csv', '10_MWp_P.csv','12_MWp_P.csv', '19_MWp_P.csv']
# mrr, em_mu, em_x, cdfs_data = \
# generate_stochastic_load(fi, fk, '15T', ['1S'], [7, 10, 12, 19], fn)

# from init import *
# from src.process_data import read_initial_data

# data = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/utils/data_generation/load/data'
# data = [data + '/7_MWp_P.csv', data + '/10_MWp_P.csv', data + '/12_MWp_P.csv', data + '/19_MWp_P.csv']
# # load_gen = LoadGenerator(data, [7, 10, 12, 19])

# #data = [data + '/7_MWp_P.csv', data + '/10_MWp_P.csv', data + '/12_MWp_P.csv']
# load_gen = LoadGenerator(data)

# a = load_gen.generate_load([7, 0.14])
# print('ahihi')


from init import *
from src.process_data import read_initial_data
data = "C:\\Users\\kathl\Desktop\\Github\\ceds-cigar\\pycigar\\utils\\data_generation\\load\\data"
data = [data + '\\7_MWp_P.csv', data + '\\10_MWp_P.csv', data + '\\12_MWp_P.csv', data + '\\19_MWp_P.csv']

dss_kw = [0.14, 0.16, 0.18, 0.07, 0.09] #assumes input in kw
load_gen = LoadGenerator(data, dss_kw) 

a = load_gen.generate_load([7, 9, 10, 12],[ .14, .16] )
