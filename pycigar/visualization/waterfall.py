
# Python program to print all paths from a source to destination.
import opendssdirect as dss  
import copy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

  
# This class represents a directed graph
# using adjacency list representation
class Graph:
  
    def __init__(self, vertices, log_dict):
        # No. of vertices
        self.V = vertices
         
        self.log_dict = log_dict
        # default dictionary to store graph
        self.graph = defaultdict(list)
        
        self.paths = [0]
        
        self.updatedPaths = []
        
        self.english = []
        
        self.line_bus_dict = {}
        
        self.line_lengths = []
        
        
  
    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
  
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''
    def printAllPathsUtil(self, u, d, visited, path):
 
        # Mark the current node as visited and store in path
        visited[u]= True
        path.append(u)
 
        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            #print(path)
            
            self.paths.append(copy.copy(path))
            
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if visited[i]== False:
                    self.printAllPathsUtil(i, d, visited, path)
                     
        # Remove current vertex from path[] and mark it as unvisited
        
       
        path.pop()
        
        visited[u]= False
        
    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):
 
        # Mark all the vertices as not visited
        visited =[False]*(self.V)
 
        # Create an array to store paths
        path = []
 
        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path)
        
        
    def cleanPaths(self):
        """
        Remove the paths from printAllPathsUtil() that are obviously not useful for waterfall diagram
        """
        
        
        valid_paths = []
        start_end_dict = {}  #key = (start_idx, end_idx), value = prev
        # dictionary to track the longest path associated with a start and end bus

        prev = 0

        for path in self.paths:

            if type(path) == int or type(path) == float or len(path) <= 1:
                # useful path      
                continue
            else: # weed out potentially non-useful paths

                key = (path[0], path[len(path) - 1])

                if key in start_end_dict.values():
                    pos = start_end_dict[key]
                    if len(valid_paths[pos]) < len(path):
                        # if it exists, check against longer entries
                        start_end_dict[key] = prev #prev = position to check against
                        valid_paths[pos] = 0
                    else:
                        continue

                if key not in start_end_dict.values():
                    # if it doesn't exist, add it to the dictionary
                    start_end_dict[key] = prev
                    
                prev += 1
                valid_paths.append(path)
      

        ls = copy.deepcopy(valid_paths)
        no_prefix = [x for x in ls if x not in [y[:len(x)] for y in ls if y != x]] #eliminate paths that are prefixes of the other
        # ex. [1, 2, 3], [1, 2], [1] -> [1, 2, 3]

        self.updatedPaths = no_prefix
    
    def convert_english(self):
        """
        convert the index associated with bus name to the bus name
        ex. 0 -> sourcebus, 2 -> 701, 3 -> 702
        convert_english([0, 2, 3]) -> ['sourcebus', '701', '702']
        """
        all_english = []
        
        for lst in self.updatedPaths:
            english = []
            for elem in lst:
                english.append(dss.Circuit.AllBusNames()[elem])
            all_english.append(english)
            
        self.english = all_english
    
    def get_line_bus_dict(self):
        """
        dictionary where
        key = (bus start, bus end)
        value = line index
        """
        line_bus_dict = {}
        
        for line in range(len(dss.Lines.AllNames())):

            dss.Lines.Name(dss.Lines.AllNames()[line])
            b1 = dss.Lines.Bus1().split('.')[0]
            b2 = dss.Lines.Bus2().split('.')[0]
            key = (b1, b2)
            line_bus_dict[key] = line
            
        line_bus_dict[('sourcebus', '701')] = 35
        self.line_bus_dict = line_bus_dict
    
    def get_line_lengths(self):
        """
        Establish the x-axis (length along the feeder)
        """
        
        line_lengths = []
        # all line lengths
        for lines in dss.Lines.AllNames():
            dss.Lines.Name(lines)
            line_lengths.append(dss.Lines.Length())   
            
        self.line_lengths = line_lengths
        
        self.line_lengths.append(0)
        
    def change_time(self, time):
        """
        Plot a different time step in the waterfall plot
        """
  
        plt.figure(figsize=(20, 10))
        plt.xlabel('Length from sourcebus (meters)')
        plt.ylabel('|V| (Voltage)')
        plt.title('Waterfall Plot')


        for j in range(len(self.updatedPaths)):
            path_lengths = [0] #list of x position
            summative = 0 #x position
            for i in range(len(self.updatedPaths[j])-1): 

                key = (self.english[j][i], self.english[j][i+1])

                summative += self.line_lengths[self.line_bus_dict[key]] 

                path_lengths.append(summative)

            v_useful = self.log_dict['v_metrics'][str(time)][0] #voltage data from log_dict at specified time step
            eng_list = self.english[j] 
            v_list = np.array([])
            for i in eng_list:
                v_list = np.append(v_list, v_useful[i]) #select voltage data relevant to segment of line (based on buses)

            y = copy.copy(v_list) #copy to avoid assignment issues (assigning to object vs label...)

            y = y.reshape(len(eng_list), 3).transpose() #differentiate between phases

            colors = {0:'orange', 1:'b', 2:'g'} #colors associated with phases

            for ii in range(len(y)):          
                plt.plot(path_lengths, y[ii], colors[ii], lw=3)


#This code is contributed by Neelam Yadavw