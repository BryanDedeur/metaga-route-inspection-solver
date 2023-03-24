import argparse
import sys
import wandb
import numpy
import time
import os

from os import path
from graph import Graph
from router import Router
from ga import DEGA

def parse_args():
    # capture the args with the parser
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-i', '--inst', dest='instance', type=str, required=True, help='filepath to problem instance')
    parser.add_argument('-k', '--k-depots', dest='k_depots', type=str, required=True, help='k num tours determined by the depots. ex: -k 0,0')
    parser.add_argument('-d', '--inverse-deployment', dest='inverse_deployment', type=bool, required=False, default=False, help='deploys the vehicles from the last vertex id minus the deployment id specified by -k. Ex n=30 -k 1,2 deploys at vertex ids 28,27')

    # parser.add_argument('-d', '--depots', dest='depots', type=str, required=True, help='the deployment configuration (single, multi). ex: -d single')
    parser.add_argument('-s', '--seeds', dest='seeds', type=str, required=True, help='random seeds to run the ga. ex: -s 1234,3949')
    parser.add_argument('--silent', dest='silent', default=False, action='store_true', help='enable silent mode')    
    args = parser.parse_args()

    # check and adjust the parsed args
    args.instance = args.instance.replace(' ', '')
    if not path.exists(args.instance):
        sys.exit("Cannot find instance: " + args.instance)
    args.k_depots = [int(i) for i in args.k_depots.split(',')]
    # args.depots = args.depots.replace(' ', '')
    args.seeds = [int(i) for i in args.seeds.split(',')]
    return args

def main():
    # capturing the arguements
    args = parse_args()

    # Define the print function
    if args.silent or os.getenv('SILENT_MODE') == '1':
        os.environ['SILENT_MODE'] = '1'

    print('Running DEGA on '+args.instance+', '+str(args.k_depots)+' depots, ' +str(len(args.seeds)) +' seeds')

    # create the graph
    gph = Graph(args.instance)

    # inverse deployment if specified
    if args.inverse_deployment:
        for i in range(len(args.k_depots)):
            args.k_depots[i] = gph.size_v() - 1 - args.k_depots[i]

    # create a router for constructing tours
    router = Router(gph, args.k_depots)

    # find the chromosome lengths based on the heuristics
    chrom_len = gph.size_e() + len(router.tours)

    # define the fitness function
    def evaluate(ga, chromosome):  
        router.clear()

        # add first vertex to tour
        for tour in router.tours:
            tour.add_vertex(tour.depot)

        # convert the heuristics to tours
        current_tour_id = -1
        for h in chromosome:
            if h > gph.size_e() - 1:
                # a vehicle
                current_tour_id = h - gph.size_e()
            else:
                # an edge
                if current_tour_id > -1:
                    router.tours[current_tour_id].add_edge(h)

        # finish the tour that was partially read
        for h in chromosome:
            if h < gph.size_e():
                router.tours[current_tour_id].add_edge(h)
            else:
                break

        # check if all edges were visited
        unvisited_edges = [i for i in range(gph.size_e())]
        for tour in router.tours:
            for edge in tour.edgeSequence:
                if edge in unvisited_edges:
                    unvisited_edges.remove(edge)

        if len(unvisited_edges) > 0:
            print("ERROR: not all edges were visited")

        # return all tours to their depots
        for tour in router.tours:
            tour.add_vertex(tour.depot)

        # compute objective
        objective = router.get_length_of_longest_tour()
        fitness = 1/objective

        # check if best
        if ga.best_fitness < fitness:
            ga.best_fitness = fitness
            ga.best_binary = numpy.copy(chromosome)
            ga.best_evaluation = ga.num_evaluations
            ga.best_generation = ga.ga_instance.generations_completed
            ga.best_time_seconds = time.time() - ga.run_time_start
            ga.best_solution = router.get_route()
            # print(ga.best_solution)

        return fitness
    
    def log_data(ga):
        wandb.log(ga.log_data)

    # create the metaga
    dega = DEGA(chrom_len, evaluate, log_data)

    for seed in args.seeds:
        router.set_seed(seed)
        dega.create(seed)
        wandb.config = {
            'ga' : dega.config,
            'instance' : gph.config,
            'routing' : router.config
        }

        wandb.init(project="metaga-data", name=gph.name +'_'+ str(seed), config=wandb.config)
        dega.run()
        wandb.log(dega.log_data)
        wandb.finish()


    # output the final results
    # print('overall best: ' + str(round(meta_ga.getOverallBestObj(),2)))
    # print('per seed average best: ' + str(round(meta_ga.getAveSeedBestObj(),2)))
    # print('per seed average num evaluations to achieve near best: ' + str(round(meta_ga.getAveNumEvalsToAveBest(),2)))
    # print('per seed reliability: ' + str(round(meta_ga.getReliability(),2)))
    # print('overall time: ' + str(round(meta_ga.seedTimeStats.sum,2)) + 's')

if __name__ == '__main__':
    main()
