import math
import ast
import pandas as pd
import numpy as np

def parser(file_name):
    df = pd.read_csv(file_name)
    df = df.set_index(list(df)[0])
    
    # get the data from file and shape it under the form:
    #initController = {
    #   5: {
    #       'type': 'AdaptiveInvController',
    #       'VBP': np.array([1.01, 1.03, 1.03, 1.05]),
    #       'nk': 0.1,
    #       'threshhold': 0.25
    #   },
    #   7: {
    #       'type': 'AdaptiveInvController',
    #       'VBP': np.array([1.01, 1.03, 1.03, 1.05]),
    #       'nk': 0.1,
    #       'threshhold': 0.25
    #   }
    
    initController = {}
    for i in range(len(df.loc['Controller'].values)):
        if df.loc['Controller'][i] is not np.nan:
            controller = {}
            controller['type'] = df.loc['Controller'][i]

            for prop in df.loc['ControllerSetting'][i].split(';'):
                controller[prop.split(':')[0].strip(' ')] = ast.literal_eval(prop.split(':')[1].strip(' '))
            initController[i] = controller


    # get the data from file and shape it under the form:
    #initHackedController = {
    #    300: {
    #        5: {
    #            'type': 'FixedInvController',
    #            'VBP': np.array([1.01, 1.015, 1.015, 1.02])
    #           },
    #    },
    #    500: {
    #        7: {
    #            'type': 'FixedInvController',
    #            'VBP': np.array([1.01, 1.015, 1.015, 1.02])
    #           }
    #       }
    #   }

    initHackedController = {}
    for i in range(len(df.loc['HackedController'].values)):
        if df.loc['HackedTimeStep'][i] is not np.nan:
            ind = ast.literal_eval(df.loc['HackedTimeStep'][i])
            hackedController = {}
            hackedController['type'] = df.loc['HackedController'][i]
            
            for prop in df.loc['HackedControllerSetting'][i].split(';'):
                hackedController[prop.split(':')[0].strip(' ')] = ast.literal_eval(prop.split(':')[1].strip(' '))

            hackedController['percentHacked'] = ast.literal_eval(df.loc['PercentHack'][i])
            if ind in initHackedController:
                initHackedController[ind][i] = hackedController
            else:
                initHackedController[ind] = {i: hackedController} 


    startTime = ast.literal_eval(df.loc['StartTime'][0])
    endTime = ast.literal_eval(df.loc['EndTime'][0])
    FileDirectoryBase = df.loc['FileDirectoryBase'][0]
    OpenDSSDirectory = df.loc['OpenDSSDirectory'][0]
    
    OpenDSSkwargs = {}
    for prop in df.loc['OpenDSSkwargs'][0].split(';'):
        OpenDSSkwargs[prop.split(':')[0].strip(' ')] = ast.literal_eval(prop.split(':')[1].strip(' '))

    
    logger_kwargs = {}
    for prop in df.loc['logger_kwargs'][0].split(';'):
        try:
            logger_kwargs[prop.split(':')[0].strip(' ')] = ast.literal_eval(prop.split(':')[1].strip(' '))
        except:
            logger_kwargs[prop.split(':')[0].strip(' ')] = prop.split(':')[1].strip(' ')

    rl_kwargs = {}

    for prop in df.loc['rl_kwargs'][0].split(';'):
        try:
            rl_kwargs[prop.split(':')[0].strip(' ')] = ast.literal_eval(prop.split(':')[1].strip(' ')) 
        except:
            rl_kwargs[prop.split(':')[0].strip(' ')] = prop.split(':')[1].strip(' ')

    args = {}
    args['initController'] = initController
    args['initHackedController'] = initHackedController
    args['startTime'] = startTime
    args['endTime'] = endTime
    args['FileDirectoryBase'] = FileDirectoryBase
    args['OpenDSSDirectory'] = OpenDSSDirectory
    args['OpenDSSkwargs'] = OpenDSSkwargs
    args['logger_kwargs'] = logger_kwargs
    #args['rl_kwargs'] = rl_kwargs

    return args