# SETUP ENV
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from pycigar.utils.input_parser import input_parser
import numpy as np
from pycigar.utils.registry import register_devcon
import tensorflow as tf
from ray.rllib.models.catalog import ModelCatalog
from gym.spaces import Tuple, Discrete, Box
import matplotlib

misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
dss = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'

start = 100
hack=0.4
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

from pycigar.utils.logging import logger
log_dict = logger().log_dict
custom_metrics = logger().custom_metrics

import pandas as pd
from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure

from bokeh.models import CategoricalColorMapper, HoverTool, ColumnDataSource, Panel
from bokeh.models.widgets import CheckboxGroup, Slider, RangeSlider, Tabs

from bokeh.layouts import column, row, WidgetBox
from bokeh.palettes import Category20_16 as palette

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
import itertools

imbalance = pd.DataFrame(log_dict['u_metrics'])*100
all_node_list = imbalance.columns.to_list()

def imbalance_plot(doc):
    def make_dataset(dataset, node_list):
        xs = []
        ys = []
        colors = []
        c = itertools.cycle(palette)
        for i, node in enumerate(node_list):
            x = dataset[node].index.to_list()
            y = dataset[node].to_list()
            xs.append(x)
            ys.append(y)
            colors.append(next(c))
        subset = pd.DataFrame({'x': xs, 'y': ys, 'color': colors, 'legend': node_list})
        return ColumnDataSource(subset)

    def make_plot(src, title='', x_axis_label='Time (s)', y_axis_label=''):
        p = figure(plot_width=600, plot_height=600, title=title, x_axis_label=x_axis_label, y_axis_label=y_axis_label)
        p.multi_line(source=src, xs='x', ys='y', line_color='color', legend_field='legend')
        return p
    
    def update(attr, old, new):
        select_node = [node_list_selection.labels[i] for i in node_list_selection.active]
        new_subset = make_dataset(imbalance, select_node)
        src.data.update(new_subset.data)

    node_list_selection = CheckboxGroup(labels=all_node_list, active=[0, 1])
    node_list_selection.on_change('active', update)
    controls = WidgetBox(node_list_selection)

    initial_nodes = [node_list_selection.labels[i] for i in node_list_selection.active]

    src = make_dataset(imbalance, initial_nodes)

    p = make_plot(src)

    layout = row(controls, p)
    doc.add_root(layout)

# Set up an application
handler = FunctionHandler(imbalance_plot)
app = Application(handler)
show(app)