import tour
import random

from graph import Graph
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D


class Router:
	def __init__(self, gph, depots, heuristics_group):
		self.graph = gph
		self.tours = []
		self.nearestEdgesSetSize = self.graph.size_e()
		self.seed = 0
		self.depots = depots
		self.heuristics_group = heuristics_group
		self.heuristics = []
		if self.heuristics_group == 'MMMR':
			gene_len = 2 # for 4 total heuristics
			self.heuristics = [            
				self.add_edges_to_shortest_tour_with_min_cost_edge_from_nearest_unvisited_equidistant, # min cost
				self.add_edges_to_shortest_tour_with_mean_cost_edge_from_nearest_unvisited_equidistant, # median cost 
				self.add_edges_to_shortest_tour_with_mean_cost_edge_from_nearest_unvisited_equidistant, # max cost
				self.add_edges_to_shortest_tour_with_random_cost_edge_from_nearest_unvisited_equidistant # random cost 
			]
		elif self.heuristics_group == 'RR':
			gene_len = len(bin(gph.maxVertexDegree)[2:]) # the binary representation of max vertex degree
			for i in range(pow(2,gene_len)): 
				self.heuristics.append(self.add_edges_to_shortest_tour_with_round_robin_nearest_unvisited_equidistant)

		for i in range(len(depots)):
			self.tours.append(tour.Tour(self.graph))
			self.tours[i].k = i
			self.tours[i].depot = self.depots[i]

		self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'maroon', 'deepskyblue', 'lime', 'gold', 'hotpink']

		self.visitedEdges = []
		self.unvisitedEdges = []

		self.config = {
			'num_tours' : len(self.tours),
			'depots' : self.depots,
			'heuristic_group' : self.heuristics_group
		}

	def data(self):
		output = {'sum costs':self.get_sum_costs()}
		for i in range(len(self.tours)):
			output['tour'+str(i)] = {'cost':self.tours[i].cost,'tour':self.tours[i].data_str()}
		return output
	
	def get_route(self):
		route = {
			"sum costs":self.get_sum_costs(),
			"tours": []
		}

		# append all the tours into the table
		for i in range(len(self.tours)):
			tour = {
				'vertices': self.tours[i].get_vertex_sequence().copy(),
				'edges': self.tours[i].get_edge_sequence().copy(),
				'cost': self.tours[i].cost
			}
			route['tours'].append(tour)

		return route
	
	def size(self):
		return len(self.tours)

	def setSeed(self, seed):
		self.seed = seed
		random.seed(seed)
		for k in range(len(self.tours)):
			self.tours[k].seed = seed

	def View(self):
		i = 0
		# fig, ax = plt.subplots()
		# sizex = 1
		# sizey = 1
		for tour in self.tours:
			axe = tour.View(i, self.colors[i])
			# fig.axes.append(axe)
			# fig.add_axes(axe)

			i += 1
		# plt.show()

		fig, ax = plt.subplots(1, figsize=(4, 4))
		ax.title.set_text('graph ' + self.graph.name.lower())
		self.ViewOverlap(ax)
		#plt.savefig(fname='img/' + self.graph.name +'-k'+str(len(self.tours))+'-'+str(self.seed) +'-overlap')
		plt.close()

	def colorFader(self, c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
		c1=np.array(mpl.colors.to_rgb(c1))
		c2=np.array(mpl.colors.to_rgb(c2))
		return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

	def ViewOverlap(self, ax):
		edgeVisits = []
		for e in range(self.graph.size_e()):
			edgeVisits.append(0)
		for tour in self.tours:
			for e in tour.edgeSequence:
				edgeVisits[e] += 1

		x = []
		y = []

		minCount = min(edgeVisits)
		maxCount = max(edgeVisits)
		minColor = 'dodgerblue'
		maxColor = 'red'
		for e in range(self.graph.size_e()):
			vpair = self.graph.get_edge_vertices(e)
			x = (self.graph.vertices[vpair[0]][0], self.graph.vertices[vpair[1]][0])
			y = (self.graph.vertices[vpair[0]][1], self.graph.vertices[vpair[1]][1])
			ax.plot(x, y, color=self.colorFader(minColor,maxColor,(edgeVisits[e] - 1)/(maxCount - minCount)), linewidth=2 * edgeVisits[e])
		legend_elements = [
			Line2D([0], [0], color=minColor, linewidth=2 * minCount, label=str(minCount) + " visits"),
			Line2D([0], [0], color=maxColor, linewidth=2 * maxCount, label=str(maxCount) + " visits")]
		#ax.legend(handles=legend_elements, loc='upper right')

	def to_string(self, delimiter = ',', ending = '\n'):
		data = [
			len(self.tours), 
			self.get_sum_costs(),
			self.getLengthOfLongestTour()
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
		# f = open(path, "a")
		# f.write(self.to_string())
		# f.close()
		self.View()

	def copy(self, other):
		self.graph = other.graph
		self.tours = []
		for t in range(len(other.tours)):
			self.tours.append(tour.Tour(self.graph))
			self.tours[t].seed = other.seed
			self.tours[t].k = len(other.tours)
		for i in range(len(self.tours)):
			for v in other.tours[i].vertexSequence:
				self.tours[i].add_vertex(v)
		self.unvisitedEdges = []
		for e in other.unvisitedEdges:
			self.unvisitedEdges.append(e)
		for e in other.visitedEdges:
			self.visitedEdges.append(e)
		self.seed = other.seed
		
	def get_sum_costs(self):
		sum = 0
		for i in self.tours:
			sum += i.cost
		return sum

	def clear(self):
		for i in range(len(self.tours)):
			self.tours[i].clear()
		self.unvisitedEdges.clear()
		self.visitedEdges.clear()
		for i in range(self.graph.size_e()):
			self.unvisitedEdges.append(i)

	def getUnvisitedEdges(self):
		return self.unvisitedEdges
	
	def getLongestTour(self):
		foundTour = None
		tempLength = 0
		for tour in self.tours:
			if tour.cost > tempLength:
				foundTour = tour
				tempLength = tour.cost
		return foundTour

	def get_shortest_tour(self):
		foundTour = None
		tempLength = float('inf')
		for tour in self.tours:
			if tour.cost < tempLength:
				foundTour = tour
				tempLength = tour.cost
		return foundTour

	def getLengthOfLongestTour(self):
		return self.getLongestTour().cost

	def get_set_of_nearest_unvisited_edges(self, vertex, maxSetSize = -1, sort = True):
		allShortestTourEdgePairs = self.getShortestToursToAllUnvisitedEdgesFromVertex(vertex)
		if len(allShortestTourEdgePairs) == 0:
			return []
		setOfEdges = []
		distanceToNearestEdges = allShortestTourEdgePairs[0][0].cost
		for tourEdgePair in allShortestTourEdgePairs:
			if maxSetSize != -1:
				# append until max set size
				if len(setOfEdges) < maxSetSize:
					setOfEdges.append(tourEdgePair[1])
				else:
					break
			else:
				# continue appending same distance edges
				if tourEdgePair[0].cost == distanceToNearestEdges:
					setOfEdges.append(tourEdgePair[1])
				else:
					break 
		def getLengthOfEdge(edgeId):
			return self.graph.get_edge_cost(edgeId)
		if sort:
			setOfEdges.sort(key=getLengthOfEdge)
		# for e in setOfEdges:
		# 	print(self.graph.get_edge_cost(e))
		return setOfEdges
	
	# --------------------------------------------------------------- TOUR CONSTRUCTING HEURISTICS ---------------------------------------------------------------------------------

	# add edges to shortest tour considering min cost from nearest unvisited equidistant set
	def add_edges_to_shortest_tour_with_min_cost_edge_from_nearest_unvisited_equidistant(self, heuristic_id : int):
		# find shortest tour last vertex
		shortest_tour = self.get_shortest_tour()
		last_vertex = shortest_tour.vertexSequence[-1]
		# find set of nearest equidistant edges
		nearest_equidistant_edges = self.get_set_of_nearest_unvisited_edges(last_vertex)
		# no more edge options
		if len(nearest_equidistant_edges) == 0:
			return -1
		# select the min cost edge
		min_cost_edge = nearest_equidistant_edges[0]
		# append all edges including selected edge
		self.extend_tour_to_edge(min_cost_edge, shortest_tour)

	# add edges to shortest tour considering mean cost from nearest unvisited equidistant set
	def add_edges_to_shortest_tour_with_mean_cost_edge_from_nearest_unvisited_equidistant(self, heuristic_id : int):
		# find shortest tour last vertex
		shortest_tour = self.get_shortest_tour()
		last_vertex = shortest_tour.vertexSequence[-1]
		# find set of nearest equidistant edges
		nearest_equidistant_edges = self.get_set_of_nearest_unvisited_edges(last_vertex)
		# no more edge options
		if len(nearest_equidistant_edges) == 0:
			return -1
		# select the min cost edge
		mean_cost_edge = nearest_equidistant_edges[len(nearest_equidistant_edges) // 2]
		# append all edges including selected edge
		self.extend_tour_to_edge(mean_cost_edge, shortest_tour)

	# add edges to shortest tour considering max cost from nearest unvisited equidistant set
	def add_edges_to_shortest_tour_with_mean_cost_edge_from_nearest_unvisited_equidistant(self, heuristic_id : int):
		# find shortest tour last vertex
		shortest_tour = self.get_shortest_tour()
		last_vertex = shortest_tour.vertexSequence[-1]
		# find set of nearest equidistant edges
		nearest_equidistant_edges = self.get_set_of_nearest_unvisited_edges(last_vertex)
		# no more edge options
		if len(nearest_equidistant_edges) == 0:
			return -1
		# select the min cost edge
		max_cost_edge = nearest_equidistant_edges[-1] 
		# append all edges including selected edge
		self.extend_tour_to_edge(max_cost_edge, shortest_tour)

	# add edges to shortest tour considering random cost from nearest unvisited equidistant set
	def add_edges_to_shortest_tour_with_random_cost_edge_from_nearest_unvisited_equidistant(self, heuristic_id : int):
		# find shortest tour last vertex
		shortest_tour = self.get_shortest_tour()
		last_vertex = shortest_tour.vertexSequence[-1]
		# find set of nearest equidistant edges
		nearest_equidistant_edges = self.get_set_of_nearest_unvisited_edges(last_vertex)
		# no more edge options
		if len(nearest_equidistant_edges) == 0:
			return -1
		# select the min cost edge
		random_cost_edge = random.choice(nearest_equidistant_edges)
		# append all edges including selected edge
		self.extend_tour_to_edge(random_cost_edge, shortest_tour)

	# add edges to shortest tour considering random cost from nearest unvisited equidistant set
	def add_edges_to_shortest_tour_with_round_robin_nearest_unvisited_equidistant(self, heuristic_id : int):
		# find shortest tour last vertex
		shortest_tour = self.get_shortest_tour()
		last_vertex = shortest_tour.vertexSequence[-1]
		# find set of nearest equidistant edges
		nearest_equidistant_edges = self.get_set_of_nearest_unvisited_edges(last_vertex)
		# no more edge options
		if len(nearest_equidistant_edges) == 0:
			return -1
		# select the min cost edge
		edge = nearest_equidistant_edges[heuristic_id % len(nearest_equidistant_edges)]
		# append all edges including selected edge
		self.extend_tour_to_edge(edge, shortest_tour)

	# --------------------------------------------------------------- END TOUR CONSTRUCTING HEURISTICS ---------------------------------------------------------------------------------

	def addVertexToTours(self, vertexId):
		for tour in self.tours:
			tour.add_vertex(vertexId)

	def addVertexToTour(self, vertexId, tour):
		tour.add_vertex(vertexId)

	def extend_tour_to_edge(self, edgeId, tour):
		if edgeId > -1:
			numEdgesInTourBeforeAddedEdges = len(tour.edgeSequence)
			tour.AddEdge(edgeId)
			
			# mark all edges along shortest path as visted
			for e in range(numEdgesInTourBeforeAddedEdges, len(tour.edgeSequence)):
				edge = tour.edgeSequence[e]
				if edge in self.unvisitedEdges:
					self.unvisitedEdges.remove(edge)
					self.visitedEdges.append(edge)

	def getShortestToursToAllUnvisitedEdgesFromVertex(self, vertex, sortTours = True):
		tempToursToEdges = []
		for e in self.getUnvisitedEdges():
			tempToursToEdges.append((self.graph.get_shortest_tour_between_vertex_and_edge(vertex, e), e))
		def getTourLengthFromPair(tourEdgePair):
			return tourEdgePair[0].cost
		if sortTours:
			tempToursToEdges.sort(key=getTourLengthFromPair)
		return tempToursToEdges

	def getShortestToursToEdgesFromVertex(self, edges, vertex, sortTours = True):
		tempToursToEdges = []
		for e in edges:
			tempToursToEdges.append((self.graph.get_shortest_tour_between_vertex_and_edge(vertex, e), e))
		def getTourLengthFromPair(tourEdgePair):
			return tourEdgePair[0].cost
		if sortTours:
			tempToursToEdges.sort(key=getTourLengthFromPair)
		return tempToursToEdges

	def buildToursWithSimpleHeuristic(self):
		# builds tours by always assigning nearest unvisited edges to the shortest tour
		self.addVertexToTours(0)

		while (len(self.getUnvisitedEdges()) > 0):
			toursToEdges = self.getShortestToursToAllUnvisitedEdgesFromVertex(self.get_shortest_tour().vertexSequence[-1])
			self.extend_tour_to_edge(toursToEdges[0][1], self.get_shortest_tour())

		self.addVertexToTours(0)


