# -*- coding: utf-8 -*-
"""
Created on May 28 2019
@author: Shammya Saha
"""

line_open = lambda line_name,terminal =1 : 'open line.' + str (line_name) + ' term=' + str(terminal)
line_close = lambda line_name,terminal =1 : 'close line.' + str (line_name) + ' term=' + str(terminal)

line_switch = {'open_line' : line_open,
               'close_line' : line_close
}