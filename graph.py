import os
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# from posixpath import split
from tour import Tour

class Graph:
    def __init__(self, filepath):
        # Define the print function
        if os.getenv('SILENT_MODE') == '1':
            def print_silent(*args, **kwargs):
                pass
            self.print = print_silent
        else:
            self.print = print
        
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.name = os.path.splitext(self.filename)[0]

        self.adjacencyMatrix = [] # [][] of edge costs
        self.edgeMatrix = [] # [][] of adjacency matrix with edge ID's instead of costs
        self.edgeIds = [] # (v1, v2)
        self.vertices = [] # (x coordiante, y coordinate)
        self.connectedEdges = [] # [vertex id] = [edges]
        self.cachedDijkstras = [] #[][] of tours
        
        self.maxVertexDegree = 0
        self.minVertexDegree = 99999999999
        self.sumVertexDegree = 0
        
        self.print('Loading graph ' + self.filename + '...', end='')
        extension = os.path.splitext(self.filename)[1]
        if extension == '.csv':
            self.load_csv(filepath)
        elif extension == '.json':
            self.load_json(filepath)
        elif extension == '.dat':
            self.load_dat(filepath)
        self.print(' Success! (vertices:' + str(self.size_v()) + ' edges:' + str(self.size_e()) + ')')


        
        if len(self.vertices) < len(self.adjacencyMatrix):
            # check for obj file
            directory = filepath.replace(self.filename, '')
            objpath = directory + self.name + '.obj'
            if (os.path.exists(objpath)):
                self.load_vertices(objpath)
            else:
                self.create_vertex_positions()
        
        self.solve_and_cache_shortest_paths()

        self.config = {
            'name' : self.name,
            'vertices' : self.size_v(),
            'edges' : self.size_e(),
            'min_vertex_degree' : self.minVertexDegree,
            'avg_vertex_degree' : self.sumVertexDegree / self.size_v(),
            'max_vertex_degree' : self.maxVertexDegree,
        }

    def create_vertex_positions(self):
        theta_distribution = np.linspace(0, 2 * np.pi, self.size_v() + 1)
        radius = 1
        a = radius * np.cos(theta_distribution)
        b = radius * np.sin(theta_distribution)
        for v in range(self.size_v()):
            self.vertices.append((a[v], b[v]))
        return

    def load_vertices(self, path):
        file = open(path, 'r')
        lines = file.readlines()
        file.close()
        for v in range(len(lines)):
            cols = lines[v].split(' ')
            if len(cols) == 4:
                self.vertices.append((float(cols[1]), float(cols[2])))
        return

    def plot(self, ax, annotate_vertices, annotate_edges):
        # plot the graph
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        # plot the edges
        for vPair in self.edgeIds:
            v1 = vPair[0]
            v2 = vPair[1]
            x = (self.vertices[v1][0], self.vertices[v2][0])
            y = (self.vertices[v1][1], self.vertices[v2][1])
            ax.plot(x, y, color="gray")
            if annotate_edges:
                xy = (x[1] + (x[0] - x[1])/2, y[1] + (y[0] - y[1])/2)
                ax.annotate(str(self.edgeMatrix[v1][v2]), xy ,ha='center',va='center', color = "white",size = 6,
                            bbox=dict(boxstyle="circle,pad=0.2", fc="gray", ec="b", lw=0))
        # plot the vertices
        x = []
        y = []
        for v in range(len(self.vertices)):
            x.append(self.vertices[v][0])
            y.append(self.vertices[v][1])
            if annotate_vertices:
                xy = (self.vertices[v][0], self.vertices[v][1])
                ax.annotate(str(v), xy ,ha='center',va='center', color = "white",size = 6,
                            bbox=dict(boxstyle="circle,pad=0.2", fc="teal", ec="b", lw=0))
    
    def view(self, blocking):
        fig, ax = plt.subplots(1, figsize=(4, 4))
        ax.title.set_text('graph ' + self.name)
        self.plot(ax, True, True)
        plt.show(block = blocking)

    def load_csv(self, file):
        file = open(file, 'r')
        lines = file.readlines()
        file.close()
        count = 0
        for r in range(len(lines)):
            cols = lines[r].split(',')
            for c in range(r, len(cols)):
                edgeCost = float(cols[c])
                if edgeCost > 0:
                    self.add_edge(r, c, edgeCost)
                    count += 1

    def load_dat(self, file):
        file = open(file, 'r')
        lines = file.readlines()
        file.close()
        for line in lines:
            bad_chars = ['(', ',', ')']
            for i in bad_chars:
                line = line.replace(i, '')
            numbers = []
            for t in line.split():
                try:
                    numbers.append(float(t))
                except ValueError:
                    pass
            if len(numbers) >= 3:
                self.add_edge(int(numbers[0] - 1), int(numbers[1] - 1), numbers[2])

    def load_json(self, file):
        file = open(file, 'r')
        data = json.load(file)
        file.close()
        #for (int e = 0; e < js["edges"].size(); ++e) {
        edgesKey = 'edges'
        vertexKey = 'vIDs'
        for e in range(len(data[edgesKey])):
            v1 = data[edgesKey][e][vertexKey][0]
            v2 = data[edgesKey][e][vertexKey][1]
            cost = float(data[edgesKey][e]['length'])
            self.add_edge(v1, v2, cost)

        # load vertice positions
        for v in range(len(data['vertices'])):
            coord = data['vertices'][v]['v2Pos']
            self.vertices.append((coord[0], 10 - coord[1]))
    
    def clear(self):
        self.adjacencyMatrix.clear()
        self.edgeMatrix.clear()
        self.edgeIds.clear()
        self.cachedDijkstras.clear()
        self.connectedEdges.clear()
        self.minVertexDegree = 999999999
        self.maxVertexDegree = 0

    def size_v(self):
        return len(self.adjacencyMatrix)

    def size_e(self):
        return len(self.edgeIds)
    
    def sum_e(self):
        sum = 0
        for e in self.edgeIds:
            sum += self.get_edge_cost(e)
        return sum

    def to_string(self, delimiter = ',', ending = '\n'):
        data = [
            self.name(),
            self.size_e(),
            self.size_v(),
            self.sum_e()
        ]
        formatted = ''
        for i in range(len(data)):
            formatted += str(data[i])
        if i < len(data) - 1:
            formatted += delimiter
        else:
            formatted += ending
        return formatted

    def save(self, path):
        f = open(path, "a")
        f.write(self.to_string())
        f.close()

    # Fixes the adjacency matrix dimenions if not the right size for the vertex id.
    def fix_dimensions(self, newLim):
        # Zero base indexing means we need to increase the counts to 1+ newLim
        newLim += 1
        # Extend the size of the matrix to match the new limit
        i = self.size_v() - 1
        while (self.size_v() < newLim):
            i += 1
            self.adjacencyMatrix.append([])
            self.cachedDijkstras.append([])
            self.edgeMatrix.append([])
        # Fix the cols
        for i in range(newLim):
            while (len(self.adjacencyMatrix[i]) < newLim):
                self.adjacencyMatrix[i].append(0)
                self.cachedDijkstras[i].append(Tour(self))
                self.edgeMatrix[i].append(-1)


    def is_valid_edge(self, v1,  v2):
        if (self.edgeMatrix[v1][v2] > -1):
            return True
        # Invalid edge
        #pr("Trying to access edge between vertices (" + v1.ToString() + " " + v2.ToString() + ") which is not valid.")
        return False

    def get_edge_vertices(self, id):
        return self.edgeIds[id]
        # for r in range(len(self.edgeMatrix)):
        #     for c in range(len(self.edgeMatrix)):        
        #         if (self.edgeMatrix[r][c] == id):
        #             return (r, c)
        # # This is dangerous but nescessary to flag issues
        # self.print("Edge id:" + str(id) + " does not exist, but trying to access it.")
        # return (-1,-1)

    def get_edge(self, v1,  v2):
        if (not self.is_valid_edge(v1, v2)):
            return -1
        return self.edgeMatrix[v1][v2]

    def get_edge_cost_from_vertices(self, v1,  v2):
        return self.adjacencyMatrix[v1][v2]
    
    def get_edge_cost(self, id):
        vertices = self.get_edge_vertices(id)
        return self.get_edge_cost_from_vertices(vertices[0], vertices[1])

    def get_opposite_vertex_on_edge(self, vertex,  edge):
        vertices = self.get_edge_vertices(edge)
        if (vertices[0] == vertex):
            return vertices[1]
        return vertices[0]
    
    def get_shortest_tour_between_vertices(self, startVertex,  endVertex):
        # pr(startVertex.ToString() + endVertex.ToString())
        tour = self.cachedDijkstras[startVertex][endVertex]
        if (tour == None):
            tour = self.cachedDijkstras[endVertex][startVertex]
        return tour
    
    def get_shortest_tour_between_vertex_and_edge(self, vertex,  edge):
        evs = self.get_edge_vertices(edge)
        tour = self.get_shortest_tour_between_vertices(vertex, evs[0])
        bestTour = tour
        if (tour.cost < bestTour.cost):
            bestTour = tour
        tour = self.get_shortest_tour_between_vertices(vertex, evs[1])
        if (tour.cost < bestTour.cost):
            bestTour = tour
        return bestTour

    def get_shortest_tour_between_edges(self, edge1,  edge2):
        e1vs = self.get_edge_vertices(edge1)
        tour1 = self.get_shortest_tour_between_vertex_and_edge(e1vs[0], edge2)
        tour2 = self.get_shortest_tour_between_vertex_and_edge(e1vs[1], edge2)
        if (tour1.cost < tour2.cost):
            return tour1
        return tour2
    
    # returns the vertex inbetween two edges
    def get_edges_connection_vertex(self, edge1,  edge2):
        vertices1 = self.get_edge_vertices(edge1)
        vertices2 = self.get_edge_vertices(edge2)
        if (vertices1[0] == vertices2[0]):
            return vertices1[0]
        elif (vertices1[0] == vertices2[1]):
            return vertices1[0]
        elif (vertices1[1] == vertices2[0]):
            return vertices1[1]
        elif (vertices1[1] == vertices2[1]):
            return vertices1[1]
        return -1

    def get_set_of_edges_connected_to_vertex(self, vertexId):
        return self.connectedEdges[vertexId]

    def get_edge_degree_at_vertex(self, vertexId):
        return len(self.get_set_of_edges_connected_to_vertex(vertexId))
    
    def get_max_vertex_degree(self):
        return self.maxVertexDegree

    def add_vertex(self, vId):
        # Make sure the adjacency matrix is the right size
        self.fix_dimensions(vId)
        # all we need to do is make sure the adjacency matrix is the right size

    # Add a new edge to the adjacency matrix
    def add_edge(self, v1,  v2,  cost):
        id = self.size_e()
        # Add vertices regarless if they exist because add vertex will resolve that
        self.add_vertex(v1)
        self.add_vertex(v2)
        # make sure edge does not already exist
        if (self.edgeMatrix[v1][v2] == -1):
            self.edgeIds.append((v1, v2))
            self.adjacencyMatrix[v1][v2] = cost
            self.adjacencyMatrix[v2][v1] = cost
            self.edgeMatrix[v1][v2] = id
            self.edgeMatrix[v2][v1] = id
        
    def min_distance(self, dist, spSet):
        best = (-1, float('inf'))
        for v in range(self.size_v()):
            if (not spSet[v] and dist[v] <= best[1]):
                best = (v, dist[v])
        if (best[0] == -1):
            self.print("No better min distance found, so returning an invalid vertex.")
        return best[0]

    def dijkstras(self, src):
        # initialization
        dist = []
        spSet = [] 
        for i in range(self.size_v()):
            dist.append(float('inf'))
            spSet.append(False)
            self.cachedDijkstras[src][i].add_vertex(src)
        dist[src] = 0

        # Find shortest paths
        for count in range(self.size_v() - 1):
            u = self.min_distance(dist, spSet)
            spSet[u] = True
            for v in range(self.size_v()):
                if not spSet[v] and self.adjacencyMatrix[u][v] > 0 and not dist[u] == float('inf') and dist[u] + self.adjacencyMatrix[u][v] < dist[v]:
                    dist[v] = dist[u] + self.adjacencyMatrix[u][v]
                    # A better tour was found, clear the existing tour
                    self.cachedDijkstras[src][v].clear()
                    # Make new tour by deep copying the best vertex sequence
                    for i in range(len(self.cachedDijkstras[src][u].vertexSequence)):
                        self.cachedDijkstras[src][v].insert_vertex(self.cachedDijkstras[src][u].vertexSequence[i])
                    self.cachedDijkstras[src][v].insert_vertex(v)
                
    def solve_and_cache_shortest_paths(self):
        """Solves Dijkstras between all pairs of vertices and stores it in the graph obj"""
        
        # solve dijkstras
        for v in range(self.size_v()):
            self.dijkstras(v)
            # store connected edges to vertex
            edges = []
            for e in range(self.size_e()):
                vertices = self.get_edge_vertices(e)
                if vertices[0] == v or vertices[1] == v:
                    edges.append(e)
            self.connectedEdges.append(edges)
            # find the max vertex degree
            if len(edges) > self.maxVertexDegree:
                self.maxVertexDegree = len(edges)
            if len(edges) < self.minVertexDegree:
                self.minVertexDegree = len(edges)
            self.sumVertexDegree += len(edges)
