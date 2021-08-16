# SETUP ENV
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from pycigar.utils.input_parser import input_parser
import numpy as np
from pycigar.utils.registry import register_devcon
import tensorflow as tf
from gym.spaces import Tuple, Discrete, Box
from pycigar.utils.logging import logger
import matplotlib
import random
import re
from waterfall import Graph

# misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
# dss = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
# load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
# breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'


misc_inputs = 'C:\\Users\\kathl\\Desktop\\Github\\ceds-cigar\\pycigar\\data\\ieee37busdata_regulator_attack\\misc_inputs.csv'
dss = 'C:\\Users\\kathl\\Desktop\\Github\\ceds-cigar\\pycigar\\data\\ieee37busdata_regulator_attack\\ieee37.dss'
load_solar = 'C:\\Users\\kathl\\Desktop\\Github\\ceds-cigar\\pycigar\\data\\ieee37busdata_regulator_attack\\load_solar_data.csv'
breakpoints = 'C:\\Users\\kathl\\Desktop\\Github\\ceds-cigar\\pycigar\\data\\ieee37busdata_regulator_attack\\breakpoints.csv'


start = 100
hack = 0.4
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=hack)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
del sim_params['attack_randomization']
for node in sim_params['scenario_config']['nodes']:
    node['devices'][0]['adversary_controller'] =  'adaptive_unbalanced_fixed_controller'

from pycigar.envs import CentralControlPhaseSpecificPVInverterEnv
env = CentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)
env.reset()
done = False
while not done:
    _, r, done, _ = env.step([10, 10, 10])

from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)

import pandas as pd
import networkx 
import copy 

from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure

from bokeh.models import CategoricalColorMapper, HoverTool, BoxSelectTool, ColumnDataSource, Panel, Column, Band, BoxAnnotation, Row, Ellipse, Legend
from bokeh.models.widgets import CheckboxGroup, CheckboxButtonGroup, Slider, RangeSlider, Tabs, MultiChoice, Div

from bokeh.layouts import column, row, WidgetBox
from bokeh.palettes import Category20_20 as palette

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
import itertools  


from bokeh.models import  CustomJS, StaticLayoutProvider, Oval, Circle, LabelSet
from bokeh.models import   GraphRenderer
from bokeh.models.widgets import  Button, DataTable, TableColumn, NumberFormatter
from bokeh.plotting import figure, from_networkx
from bokeh.models.graphs import  NodesAndLinkedEdges, EdgesAndLinkedNodes, NodesOnly


from bokeh.models import Range1d, Circle, MultiLine
from bokeh.plotting import from_networkx
from bokeh.palettes import  Oranges, RdBu

import opendssdirect as dss

def whole_plot(log_dict, inverter_list, adversary_inv, network, track_colors,cutoffs,redblue):
    
    node_end = []
    # node list like ['s701a', 's702b']
    for i in inverter_list:
        temp = i.split('_')[1]
        node_end.append(temp)
    node_end.append('y_mean')
    node_end.append('y_worst')


    g = Graph(len(dss.Circuit.AllBusNames()), log_dict)

    def line_bus_dict():
        # dict
        # key = line name 
        # value = [bus1, bus2]
        lb = {}
        for line in dss.Lines.AllNames():
            dss.Lines.Name(line)
            bus1 = dss.Lines.Bus1().split(".")[0]
            bus2 = dss.Lines.Bus2().split(".")[0]
            lb[line] = [bus1, bus2]
            
        return lb

    line_bus_dict = line_bus_dict()

    for k in line_bus_dict.keys():
        line_idx = dss.Lines.AllNames().index(k)
        bus_idxs = line_bus_dict[k] 
        bus_idx1 = dss.Circuit.AllBusNames().index(bus_idxs[0])
        bus_idx2 = dss.Circuit.AllBusNames().index(bus_idxs[1])
        g.addEdge(bus_idx1, bus_idx2)
        
    g.addEdge(0, 2) #connect sourcebus and 701

    s = 0
    d = len(dss.Circuit.AllBusNames())
    #print("Following are all different paths from % d to % d :" %(s, d-1))

    for s in range(0,1):
        for d in range(0, d):  
            g.printAllPaths(s, d)
            
            
    g.cleanPaths()

    g.convert_english()

    g.get_line_lengths()

    g.get_line_bus_dict()

    all_english = g.english

    line_lengths = g.line_lengths

    # g.change_time(400)
        
    
    ######################################
    # Y
    
    # store y data
    
    # y for inverter
    def y_data():
        df = pd.DataFrame()
        for inv in inverter_list:
            df = df.append(pd.Series(log_dict[inv]['y'], name=inv))

        tags = ['y_worst_node', 'y_worst', 'y_mean']
        for tag in tags:
            df = df.append(pd.Series(data=log_dict['y_metrics'][tag], name=tag)) 

        y_df = df.transpose()

        # y for adversary
        adv_y_df = pd.DataFrame()
        for inv in adversary_inv:
            adv_y_df = adv_y_df.append(pd.Series(log_dict[inv]['y'], name=inv))

        adv_y_df = adv_y_df.transpose()

        # combine y data
        all_y_df = pd.concat([y_df, adv_y_df], axis=1)

        imbalance_df = all_y_df*100

        # color palette
        c = itertools.cycle(palette)
        colors = {}
        for node in node_end:
            colors[node] = next(c)
        return imbalance_df, colors

    
    imbalance_df, colors = y_data()
    
 
    def make_y_dataset(dataset, node_list, range_start=0, range_end=650):
        
             
        updated_dataset = pd.DataFrame()
        
        # select marked nodes
        for node in node_list:
            
            updated_dataset = pd.concat([updated_dataset, dataset.loc[:, [node in s for s in dataset.columns.values]]], axis=1)
        
        # select y worst and y mean
        updated_dataset = pd.concat([updated_dataset, dataset['y_worst']], axis=1)
        
        updated_dataset = pd.concat([updated_dataset, dataset['y_mean']], axis=1)
        
        return ColumnDataSource(updated_dataset.loc[:, :])

    def make_y_plot(src, nodes, a_d, y_selection, title='Oscillation Intensity', x_axis_label='Time (s)', y_axis_label='Y Metric'):      

        p = figure(plot_width=800, plot_height=300,
                 title=title,x_axis_label=x_axis_label, y_axis_label=y_axis_label)
        legend_labels = []
        
        for i in y_selection:
            # for y_worst and y_mean
            y_select = y_box.labels[i]

            glyph = p.line(source=src, x='index', y=y_select,  color=colors[y_select], line_width=2) #legend_label=node, for all if statements
            hover = HoverTool(renderers=[glyph], tooltips=[('{}'.format(y_select), '$y')])
            p.add_tools(hover)
            
            legend_labels.append((y_select, [glyph])) 
                
                
        for node in nodes:
            # for y at a specific inverter
            
            if 0 in a_d: # attacker/adversary checked
                glyph = p.line(source=src, x='index', y="inverter_"+ node,  color=colors[node], line_width=2)
                hover = HoverTool(renderers=[glyph], tooltips=[('adversary_inverter_{}'.format(node), '$y')])
                p.add_tools(hover)
            elif 1 in a_d: # defender
                glyph = p.line(source=src, x='index', y="inverter_"+ node, color=colors[node], line_width=2)
                hover = HoverTool(renderers=[glyph], tooltips=[('inverter_{}'.format(node), '$y')])
                p.add_tools(hover)
                
            legend_labels.append((node, [glyph])) 

        
        legend = Legend(items=legend_labels)
        p.add_layout(legend, 'right')
            
        return p
    
    
    ######################################
    # U 
    
    # get u data
    u_df = pd.DataFrame(log_dict['u_metrics'])*100
    all_node_list = u_df.columns.to_list()

    u_df['u_std_lower'] = u_df['u_mean'] - u_df['u_std']
    u_df['u_std_upper'] = u_df['u_mean'] + u_df['u_std']
    

    def make_u_dataset(dataset, nodes, u_selection, range_start=0, range_end=650):
        # get number portion 's701a' -> '701'
        nodes = [re.findall("\d+", i)[0] for i in nodes]
     
        # get u_std
        if u_selection:
            nodes.append('u_std_lower')
            nodes.append('u_std_upper')
            
        return ColumnDataSource(dataset[nodes][:])
    
    def make_u_plot(src, nodes, u_selection, title='Oscillation Imbalance', x_axis_label='Time (s)', y_axis_label='U Metric'):
       
    
        clipped_node = [re.findall("\d+", i)[0] for i in nodes]
        clipped_node = list(set(clipped_node))
       
        p = figure(plot_width=800, plot_height=300, 
                   title=title,
                   x_range=(src.data['index'][0], src.data['index'][-1]), y_range=(0, 6),
                   x_axis_label=x_axis_label, y_axis_label=y_axis_label)
        legend_labels = []
        
        if u_selection:  
            band = Band(source=src, base='index', lower='u_std_lower', upper='u_std_upper', 
            level='underlay', fill_alpha=0.5, line_width=0.5, fill_color='green')
            p.add_layout(band)
   
        for node in range(len(clipped_node)):
            
            glyph = p.line(source=src, x='index', y=clipped_node[node],  color=colors[nodes[node]], line_width=2) 
            hover = HoverTool(renderers=[glyph], tooltips=[('{}'.format(clipped_node[node]), "$y")])
            p.add_tools(hover)
            legend_labels.append((clipped_node[node], [glyph])) # set up legend labels for display outside plot
        
        legend = Legend(items=legend_labels)
        p.add_layout(legend, 'right') # add legend
            

        # bands of color in graph
#         high_box = BoxAnnotation(bottom=6, fill_alpha=0.05, fill_color='red')
#         mid_box = BoxAnnotation(bottom=3, top=6, fill_alpha=0.05, fill_color='orange')
#         p.add_layout(high_box)
#         p.add_layout(mid_box)
        
        
        return p
    
    ######################################
    # Tap
    
    def make_tap_dataset(regname=['reg1'], range_start=0, range_end=650):
        
        x = []
        y = []
        
        for reg in regname:
            
            yy = log_dict[reg]['tap_number']
            xx = np.arange(len(yy))
            
            x.append(xx)
            y.append(yy)
            
        return x, y

    def make_tap_plot(x, y, regname, title='Tap Number Changes', x_axis_label='Time (s)', y_axis_label='Tap Number'):

        p = figure(plot_width=800, plot_height=300, 
                   title=title,
                   x_axis_label=x_axis_label, y_axis_label=y_axis_label)   
        legend_labels = []
        for i in range(len(x)):

            glyph = p.line(x=x[i], y=y[i], line_width=2)
            hover = HoverTool(renderers=[glyph], tooltips=[("Tap Number", "$y")])
            p.add_tools(hover)
            
            legend_labels.append((regname[i], [glyph])) # set up legend labels for display outside plot
        
        
        legend = Legend(items=legend_labels)
        p.add_layout(legend, 'right') # add legend

        return p
    
    ######################################
    # Voltage

    def generate_voltage_df(phase):
        # phase - base-0

        v_df = pd.DataFrame()

        for val in log_dict['v_metrics'].values():

            for i in val:

                vals = np.array(list(i.values()))
                key = np.array(list(i.keys()))

                temp = pd.DataFrame(data=vals[:, phase], index=key)
                v_df = pd.concat([v_df, temp], axis = 1)

        v_a = v_df.transpose().reset_index().drop(labels='index', axis=1)

        return v_a  


    def make_voltage_dataset(nodes, select_stats):
        
        nodes = [re.findall("\d+", i)[0] for i in nodes]
        
        v_a = generate_voltage_df(0)
        v_b = generate_voltage_df(1)
        v_c = generate_voltage_df(2)

        all_voltages = [v_a, v_b, v_c]
        all_node_list = v_c.columns

        dataset = []
        

        for i in range(len(select_stats)):
            dataset.append(all_voltages[int(select_stats[i])].loc[:, nodes])           
        
        return dataset
    
    def make_voltage_plot(src, nodes, phases, title='Bus Voltage', x_axis_label='Time (s)', y_axis_label='Voltage (V)'):
        
        clipped_nodes = [re.findall("\d+", i)[0] for i in nodes]
        
       
        p = figure(plot_width=800, plot_height=300, 
                title=title,
                x_axis_label=x_axis_label, y_axis_label=y_axis_label)    


        p.xaxis.major_label_orientation = "vertical" # alternatively, math.pi/2

        color_dict = {0: 'A', 1:'B', 2:'C'}
        legend_it = []
        clipped_nodes = list(set(clipped_nodes))
        for s in range(len(phases)):
            for node in range(len(clipped_nodes)):
                
              
                subset = src[s].iloc[:, node]
                
                # s + node name + phase =>  s 701 a
                pos = "s" + str(clipped_nodes[node]) + color_dict[phases[s]].lower()
                c = itertools.cycle(palette)
                if pos not in colors.keys():
                    c_store = next(c)
                    while c_store not in colors.values():
                   
                       c_store = next(c)
                    colors[pos] = next(c)
              
                glyph = p.line(y=list(subset.values), x=list(subset.index), color=colors[pos], line_width=2)  
                

                legend_it.append((pos, [glyph]))
              

                hover = HoverTool(renderers=[glyph], tooltips=[(clipped_nodes[node], 'Phase {}'.format(color_dict[s])),("Voltage", "$y")])
                p.add_tools(hover)
                
               

        legend = Legend(items=legend_it)
        p.add_layout(legend, 'right') # add legend

        return p
    
    ######################################
    # Control Setting
    
    
    ##### Functions
    def get_control_params(k):
        x = np.array(log_dict[k]['control_setting'])
        y = pd.DataFrame(data=x, columns=['control1', 'control2', 'control3', 'control4', 'control5'])
        y = y.transpose()

        return y

    def get_inv_control_dict(inv_data):
        inv_control_dict = {}
        for k in inv_data:
            inv_control_dict[k] = get_control_params(k)
        return inv_control_dict

    def get_translation_and_slope(a_val, init_a):
        points = np.array(a_val)
        slope = points[:, 1] - points[:, 0]
        translation = points[:, 2] - init_a[2]
        return translation, slope

    def all_inv_translation_and_slope(inv_data):
        t_s_dict = {}
        for inv in inv_data:
                a_val = logger().log_dict[inv]['control_setting']
                init_a = logger().custom_metrics['init_control_settings'][inv]
                t, s = get_translation_and_slope(a_val, init_a)
                t_s_dict[inv] = [t, s]
        return t_s_dict

    ##### 


    inv_control_dict = get_inv_control_dict(adversary_inv)
    inv_control_dict2 = get_inv_control_dict(inverter_list)

    t_s_dict = all_inv_translation_and_slope(adversary_inv)
    t_s_dict2 = all_inv_translation_and_slope(inverter_list)
    
    
    
    def make_control_dataset(inv):
        
        # select control settings for checked inverters
        src = {}
        for i in inv:
            pfix2 = 'inverter_' + str(i)
            pfix = 'adversary_inverter_' + str(i)
            src[i] = [t_s_dict[pfix], t_s_dict2[pfix2]]

        return src

    def make_control_plot(src, inverter_type, geometry_type, title='Control Setting', x_axis_label='Time (s)', y_axis_label=''):
        legend_labels = []
        tag = 0

    
        select_tools = ['box_select', 'lasso_select', 'poly_select', 'tap', 'reset']


        left = figure(plot_width=800, 
                      plot_height=500, 
                      title=title, 
                      x_axis_label=x_axis_label, 
                      y_axis_label=y_axis_label, 
                      toolbar_location='right',
                      tools=select_tools)
   

        for i in src.keys():

            dim = len(src[i][0][0])
            x = list(np.linspace(0, dim, dim))
            tags = {1: 'inverter', 0:'adversary'}

            for j in range(2): #inverter [1] vs adversary [0]

                for k in range(2): #translation [0] vs slope [1]

                    if j in inverter_type.active and k in geometry_type.active: # plot if selected

                        cds = ColumnDataSource(data=dict(x=x, y0=src[i][j][k])) # format data for plotting

                        p = left.line('x', 'y0',  hover_color="firebrick", source=cds, color=colors[i], 
                                      line_width=3)
                        
                    
                        # hover text based on slope or translation
                        if k == 1:
                            #p.glyph.line_dash = 'dashed'
                            hover = HoverTool(renderers=[p], tooltips=[("slope", "$y"), (tags[j], i)])
                            left.add_tools(hover)

                        else:
                            #p.glyph.line_dash = 'solid'
                            hover = HoverTool(renderers=[p], tooltips=[("translation", "$y"), (tags[j], i)])
                            left.add_tools(hover)  
                        tag = 1
            if tag == 1:              
                legend_labels.append((i, [p])) # set up legend labels for display outside plot

        if tag == 1:
            legend = Legend(items=legend_labels)
            left.add_layout(legend, 'right') # add legend
        tag = 0

        
        return left
    
    ######################################
    # PQ 
        
    def make_pq_dataset( node_list, inverter_type, range_start=0, range_end=699):
        
        inv_all = [i for i in log_dict.keys() if 'inverter_' in i]
        df_p_old = {}
        df_q_old = {}
        df_p_new = {}
        df_q_new = {}
        for inv in inv_all:
            df_p_old[inv] = log_dict[inv]['p_out']
            df_p_new[inv] = log_dict[inv]['p_out_new']
            df_q_old[inv] = log_dict[inv]['q_out_new']
            df_q_new[inv] = log_dict[inv]['q_out']

        df_p_old = pd.DataFrame(df_p_old)
        df_p_new = pd.DataFrame(df_p_new)
        df_q_old = pd.DataFrame(df_q_old)
        df_q_new = pd.DataFrame(df_q_new)

        df = [df_p_old, df_q_old, df_p_new, df_q_new]

        for d in df:
            d['total'] = d.sum(axis=1)


        inv_all.append('total')
          
        node_list_new = []
        for j in inverter_type.active:
            if j == 0: #0 is adversary
                lst = ['adversary_inverter_' + node for node in node_list]
                
                node_list_new.append(lst)
            if j == 1:
                lst = ['inverter_' + node for node in node_list]
                
                node_list_new.append(lst)
       
        return [ColumnDataSource(df[i].loc[:, node_list_new[0]]) for i in range(len(df))]

    def make_pq_plot(src, nodes, inverter_type, title='Real and Reactive Power (PQ)', x_axis_label='Time (s)', y_axis_label=''):
        
        p = figure(plot_width=800, plot_height=400, 
                   title=title,
                   x_range=(src[0].data['index'][0], src[0].data['index'][-1]),
                   x_axis_label=x_axis_label, y_axis_label=y_axis_label)
        
        legend_labels = []
        for node in nodes:
            if 1 in inverter_type.active:
                
                #starting_color = itertools.islice(itertools.cycle(palette), pos[inv]+random.randint(1, 6), None)
                glyph1 = p.line(source=src[0], x='index', y='inverter_' + node, color=colors[node], 
                               line_width=1)
                glyph2 = p.line(source=src[1], x='index', y='inverter_' +node, color=colors[node], 
                               line_width=2)
                glyph3 = p.line(source=src[2], x='index', y='inverter_' +node, color=colors[node], 
                                line_width=1)
                glyph4 = p.line(source=src[3], x='index', y='inverter_' +node, color=colors[node],
                               line_width=2)
                
                glyphs = [glyph1, glyph2, glyph3, glyph4]
                tag = ['p_old_', 'p_new_', 'q_old_', 'q_new_']
                
                
                for g in range(len(glyphs)):
                    legend_labels.append(( 'inv_' + tag[g] + node[-4:], [glyphs[g]])) 
                
            if 0 in inverter_type.active: # 0 is adversary
                #starting_color = itertools.islice(itertools.cycle(palette), pos[inv]+random.randint(1, 6), None)
                glyph1 = p.line(source=src[0], x='index', y='adversary_inverter_' + node,  color=colors[node], 
                                 line_width=1)
                glyph2 = p.line(source=src[1], x='index', y='adversary_inverter_' +node,color=colors[node],  
                                line_width=2)
                glyph3 = p.line(source=src[2], x='index', y='adversary_inverter_' +node, color=colors[node], 
                                line_width=1)
                glyph4 = p.line(source=src[3], x='index', y='adversary_inverter_' +node,color=colors[node], 
                                line_width=2)
                
                glyphs = [glyph1, glyph2, glyph3, glyph4]
                tag = ['p_old_', 'p_new_', 'q_old_', 'q_new_']

                for g in range(len(glyphs)):
                    legend_labels.append(( 'adv_' + tag[g] + node[-4:], [glyphs[g]])) 
            
            
            #pretag = {0:'adv_', 1:'inv_'}
            
            for i in range(len(glyphs)):               
                hover = HoverTool(renderers=[glyphs[i]], tooltips=[(tag[i] + "{}".format(node), '{}'.format('$y'))])
                p.add_tools(hover)

 
            
        legend = Legend(items=legend_labels)
        p.add_layout(legend, 'right') # add legend

        return p

    ######################################
    # waterfall 

    def make_waterfall_plot(waterfall_time, title='Waterfall Plot', x_axis_label='Distance from source (m)', y_axis_label='Voltage (V)'):
        legend_labels = []
        tag = 0

    
        select_tools = ['box_select', 'lasso_select', 'poly_select', 'tap', 'reset']


        wfall = figure(plot_width=800, 
                      plot_height=500, 
                      title=title, 
                      x_axis_label=x_axis_label, 
                      y_axis_label=y_axis_label, 
                      toolbar_location='right',
                      tools=select_tools)
        

   
        for j in range(len(g.updatedPaths)):
            path_lengths = [0]
            summative = 0
            for i in range(len(g.updatedPaths[j])-1): 

                key = (g.english[j][i], g.english[j][i+1])

                summative += g.line_lengths[g.line_bus_dict[key]]
                path_lengths.append(summative)

            v_useful = log_dict['v_metrics'][str(waterfall_time)][0]
            eng_list = g.english[j]
            v_list = np.array([])
            for i in eng_list:
                v_list = np.append(v_list, v_useful[i])

            y = copy.copy(v_list)

            y = y.reshape( len(eng_list), 3).transpose()

            colors = {0:'orange', 1:'blue', 2:'green'}

            for ii in range(len(y)):          
                #plt.plot(path_lengths, y[ii], colors[ii], lw=3)
        
                cds = ColumnDataSource(data=dict(x=path_lengths, y0=y[ii])) # format data for plotting

                p = wfall.line('x', 'y0',  hover_color="firebrick", source=cds, color=colors[ii], 
                              line_width=3)

#         hover = HoverTool(renderers=[p], tooltips=[("slope", "$y"), (tags[j], i)])
#         wfall.add_tools(hover)

             
        
#             if tag == 1:              
#                 legend_labels.append((i, [p])) # set up legend labels for display outside plot

#         if tag == 1:
#             legend = Legend(items=legend_labels)
#             left.add_layout(legend, 'right') # add legend
#         tag = 0

        
        return wfall
    
    ######################################
    # When new portion of graph selected
    global select_node_temp # stores the previous selection to prevent it from wiping the graphs when buttons are clicked
    select_node_temp = ['s701a', 's701b']
    
    def update(attr, old, new):
        global select_node_temp
      
        # process graph selection
        selected_idx = graph.node_renderer.data_source.selected.indices
        graph_node_list= []
        for i in selected_idx:
            graph_node = graph.node_renderer.data_source.data['index'][i]
            
            for col in imbalance_df.columns:
                if str(graph_node) in col:
                    j_clean = col.split('_')[len(col.split('_')) - 1]
                    graph_node_list.append(j_clean)
  
        select_node = list(set(graph_node_list)) #selected nodes
    
        
        if select_node == []:
            select_node = select_node_temp
        else:
            select_node_temp = select_node
       

        if 0 in figs_to_show.active:
        # voltage
            new_subset = make_voltage_dataset(select_node, stats_selection.active)  
            layout.children[2] = Row(stats_selection,make_voltage_plot(new_subset, select_node, stats_selection.active))
        else:
            p = figure(plot_width=10, plot_height=10)
            layout.children[2] = p

        if 1 in figs_to_show.active:
        # control setting
            new_subset = make_control_dataset(select_node)
            layout.children[3] = Row(geometry_type, make_control_plot(new_subset, inverter_type, geometry_type))
        else:
            p = figure(plot_width=10, plot_height=10)
            layout.children[3] = p
        
        if 2 in figs_to_show.active:
        # y
            new_subset = make_y_dataset(imbalance_df, select_node)
            layout.children[4] = Row(y_box, make_y_plot(new_subset, select_node, inverter_type.active, y_box.active)) #,y_box.active 
        else:
            p = figure(plot_width=10, plot_height=10)
            layout.children[4] = p

        if 3 in figs_to_show.active:
            # u
            src2 = make_u_dataset(u_df, select_node, u_std.active)    #u_std.active 
            layout.children[5] = Row(u_std, make_u_plot(src2, select_node, u_std.active)) #,u_std.active 
        else:
            p = figure(plot_width=10, plot_height=10)
            layout.children[5] = p

        if 4 in figs_to_show.active:
            
            tapx, tapy = make_tap_dataset(['reg1'])
            layout.children[6] = make_tap_plot(tapx, tapy, ['reg1'])
        else:
            p = figure(plot_width=10, plot_height=10)
            layout.children[6] = p
        
        # pq
        if 5 in figs_to_show.active:
            new_subset = make_pq_dataset(select_node, inverter_type)
            layout.children[7] = make_pq_plot(new_subset, select_node, inverter_type)
        else:
            p = figure(plot_width=10, plot_height=10)
            layout.children[7] = p

        # waterfall
        if 6 in figs_to_show.active:
            p = make_waterfall_plot(wfall_slider.value)
            layout.children[8] = p
            
            layout.children[9] = wfall_slider
        else:
            p = figure(plot_width=10, plot_height=10)
            layout.children[8] = p
            
            layout.children[9] = p
        
    ######################################
    # Button/Select

    # adversary/defender button
    inv_LABELS = ["adversary", "defender"]
    inverter_type = CheckboxButtonGroup(labels=inv_LABELS, active=[0], width=150)
    inverter_type.on_change('active', update)
    
    # initial node selection
    node_list_selection = MultiChoice(value=['s701a', 's701b'], options=node_end)
    initial_nodes = node_list_selection.value  

    # voltage  
    LABELS = ["A", "B", "C"]
    stats_selection = CheckboxButtonGroup(labels=LABELS, active=[0, 1], width=100)
    stats_selection.on_change('active', update)

    # translation/slope button
    geometry_LABELS = ["translation", "slope"]
    geometry_type = CheckboxButtonGroup(labels=geometry_LABELS, active=[0], width=150)
    geometry_type.on_change('active', update)
    
    # u std band
    u_labels = ["u_std"]
    u_std = CheckboxButtonGroup(labels=u_labels, active=[0], width=150)
    u_std.on_change('active', update)
    
    # y worst/y mean
    y_labels = ["y_worst", "y_mean"]
    y_box = CheckboxButtonGroup(labels=y_labels, active=[0], width=150)
    y_box.on_change('active', update)
    
    #waterfall slider   
    result = list(log_dict['v_metrics'].keys())
    results = list(map(int, result))
    v_start = min(results)
    v_end = max(results)
    wfall_slider = Slider(start=v_start, end=v_end, value=v_start, step=1, title="Time step")
    wfall_slider.on_change('value', update)

    
    # toggle figures
    figs_to_show_labels = ['V', 'VVC', 'Y', 'U', 'Tap', 'PQ', 'Waterfall']
    figs_to_show = CheckboxButtonGroup(labels=figs_to_show_labels, active=[0, 1, 2, 3, 4, 5, 6], width=800, align='center')
    figs_to_show.on_change('active', update)
    
    
   
    ######################################
    # Initial figures/datasets
    
    src = make_y_dataset(imbalance_df, initial_nodes)  
    p = make_y_plot(src, initial_nodes, inverter_type.active, y_box.active)
    
    src2 = make_u_dataset(u_df, initial_nodes, u_std.active)
    p2 = make_u_plot(src2, initial_nodes, u_std.active)
    
    src3 = make_voltage_dataset(initial_nodes, stats_selection.active)
    p3 = make_voltage_plot(src3, initial_nodes, stats_selection.active)
 
    src4 = make_control_dataset(initial_nodes)
    p4 = make_control_plot(src4, inverter_type, geometry_type)   
    
    tapx, tapy = make_tap_dataset(['reg1'])
    p_tap = make_tap_plot(tapx, tapy, ['reg1'])

    src5 = make_pq_dataset( initial_nodes, inverter_type)
    p5 = make_pq_plot(src5, initial_nodes, inverter_type)

    waterfall_time = v_start
    p6 = make_waterfall_plot(waterfall_time)
   

    ######################################
    # Network/Graph 
    
    G = networkx.from_pandas_edgelist(network, 'Source', 'Target', 'Weight') 
    G_source = from_networkx(G, networkx.spring_layout, scale=2, center=(0,0))
   

    graph = GraphRenderer()
    node_name = list(G.nodes())
   
    positions = networkx.spring_layout(G, seed=2)
   
    
    def selected_points(attr,old,new):       
        selected_idx = graph.node_renderer.data_source.selected.indices        
        graph.node_renderer.data_source.selected.on_change("indices", update)
         
    
    plot = figure(title = "IEEE 37 Bus Network",
                  tools = "pan, wheel_zoom, box_select, lasso_select, reset, tap", 
                  plot_width = 800, plot_height = 300, 
                  active_drag = "box_select", align='center')

    
    # Assign layout for nodes, render graph, and add hover tool
    graph.layout_provider = StaticLayoutProvider(graph_layout=positions)    
    graph.node_renderer.glyph = Ellipse(height=0.05, width=0.025, fill_color="fill_color")

    graph.selection_policy = NodesOnly()
    
    # Graph Labels
    xx, yy = zip(*graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'xx': xx, 'yy': yy, 'name': [node_labels[i] for i in range(len(xx))]})
    labels = LabelSet(x='xx', y='yy', text='name', 
                      source=source, background_fill_color='white', text_color='blue',
                      text_font_size='10px', background_fill_alpha=.7)
    
    plot.renderers.append(labels)  
    plot.renderers.append(graph)
    
    plot.tools.append(HoverTool(tooltips=[('Bus Node', '@index')]))
    
    graph.node_renderer.data_source.selected.on_change("indices", selected_points)
    
    sub_G = networkx.from_pandas_edgelist(network, 'Source', 'Target', 'Weight')
    sub_graph = from_networkx(sub_G, networkx.spring_layout, scale=2, center=(0,0))     
    graph.edge_renderer.data_source.data = dict(sub_graph.edge_renderer.data_source.data)
    graph.node_renderer.data_source.data = dict(G_source.node_renderer.data_source.data, fill_color=track_colors[:])   #was 1:  
    
     ### network L E G E N D
   
    data = dict(types = np.round(cutoffs, 4), color = redblue)

    ite = []
    
    for c in range(len(cutoffs)):
        r0 = plot.square([0], [0], fill_color=redblue[c], line_color='white', size = 0)
        ite.append(("V < " + str(np.round(cutoffs[c], 4)), [r0]))
        
    legend = Legend(items=ite, location="center")

    plot.add_layout(legend, 'right')
    
   

    
    ################ 
    
    
    div = Div(text=""" <br><br> """, width=800, height=800)
    
    layout = Column(Column(plot, div, figs_to_show), inverter_type, Row(stats_selection, p3), 
                    Row(geometry_type, p4), Row(y_box, p),  Row(u_std, p2), p_tap,  p5, p6, wfall_slider)
    return layout


class NetworkVis:
    
    def __init__(self, log_dict, dss_file):
        
        dss.run_command('Redirect ' + dss_file)
        self.dss_file = dss_file
        self.log_dict = log_dict
    

        return None
      
    def get_inverter_list(self):
        
        self.inverter_list = []
        self.adversary_inv = []
        for i in self.log_dict.keys():
            if 'inverter_' in i and 'adversary' not in i:
                self.inverter_list.append(i)
            elif 'adversary' in i:
                self.adversary_inv.append(i)
                

    def build_network(self):

        bus1 = []
        bus2 = []
        for line in range(len(dss.Lines.AllNames())):
            dss.Lines.Name(dss.Lines.AllNames()[line])   
            bus1.append((dss.Lines.Bus1().split('.')[0]))
            bus2.append((dss.Lines.Bus2().split('.')[0]))
        bus1.append('sourcebus')
        bus2.append('799')
        bus1.append('799')
        bus2.append('799r')

        self.network = pd.DataFrame({'Source': bus1, 'Target':bus2, 'Weight':np.ones(len(bus2))})
        
        
    
    def voltage_colors(self):
        
        def generate_voltage_df(phase):
           
            v_df = pd.DataFrame()

            for val in self.log_dict['v_metrics'].values():

                for i in val:

                    vals = np.array(list(i.values()))
                    key = np.array(list(i.keys()))

                    temp = pd.DataFrame(data=vals[:, phase], index=key)
                    v_df = pd.concat([v_df, temp], axis = 1)

            v_a = v_df.transpose().reset_index().drop(labels='index', axis=1)

            return v_a  

        v = generate_voltage_df(0)
        
        self.track_colors = []
        track_colors_dict = {}

        self.redblue = list(RdBu[11])

        hist = np.histogram(v.mean())
        occur = hist[0]
        self.cutoffs = hist[1]

        
        indexes = [i for i in range(len(occur)) if occur[i] == 0]
        
        for index in sorted(indexes, reverse=True):
            self.cutoffs = np.delete(self.cutoffs, index)
            del self.redblue[index]
            


        for i in range(len(v.mean())):
            for j in range(len(self.cutoffs)):
                if v.mean()[i] <= self.cutoffs[j]:
                    self.track_colors.append(self.redblue[j])
                    track_colors_dict[v.mean().index[i]] = self.redblue[j]
                    break
      
       

    def make_plot(self):  
        
        self.get_inverter_list()
        self.build_network()
        self.voltage_colors()
        
        
        def total_app(doc):
            
            layout = whole_plot(self.log_dict, self.inverter_list, self.adversary_inv, self.network, 
                                self.track_colors, self.cutoffs, self.redblue)
            doc.add_root(layout)

        handler = FunctionHandler(total_app)
        app = Application(handler)
        show(app)
        
        return None
    