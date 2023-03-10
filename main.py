import argparse
import sys
import wandb
import numpy
import time

from os import path
from graph import Graph
from router import Router
from ga import MetaGA

def parse_args():
    # capture the args with the parser
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-i', '--inst', dest='instance', type=str, required=True, help='filepath to problem instance')
    parser.add_argument('-k', '--k-depots', dest='k_depots', type=str, required=True, help='k num tours determined by the depots. ex: -k 0,0')
    # parser.add_argument('-d', '--depots', dest='depots', type=str, required=True, help='the deployment configuration (single, multi). ex: -d single')
    parser.add_argument('-s', '--seeds', dest='seeds', type=str, required=True, help='random seeds to run the ga. ex: -s 1234,3949')
    parser.add_argument('-j', '--heuristics', dest='heuristics', type=str, default='MIN-MIN', required=False, help='the set of heuristics (MIN-MIN, MIN-MAX, MIN-RANDOM). ex: -j MMMR')
    args = parser.parse_args()

    # check and adjust the parsed args
    args.instance = args.instance.replace(' ', '')
    if not path.exists(args.instance):
        sys.exit("Cannot find instance: " + args.instance)
    args.k_depots = [int(i) for i in args.k_depots.split(',')]
    # args.depots = args.depots.replace(' ', '')
    args.seeds = [int(i) for i in args.seeds.split(',')]
    args.heuristics = args.heuristics.replace(' ', '')
    return args

def main():
    # capturing the arguements
    args = parse_args()

    # create the graph
    gph = Graph(args.instance)

    # find the chromosome lengths based on the heuristics
    chrom_len = gph.size_e()

    heuristic_id = 0
    
    for seed in args.seeds:
        # create a router for constructing tours
        router = Router(gph, args.k_depots, args.heuristics)
        router.set_seed(seed)

        # define the fitness function
        def evaluate():
            router.clear()

            # add first vertex to tour
            for tour in router.tours:
                tour.add_vertex(tour.depot)

            # convert the heuristics to tours
            for h in range(chrom_len):
                router.heuristics[heuristic_id](heuristic_id)

            # return all tours to their depots
            for tour in router.tours:
                tour.add_vertex(tour.depot)

            # compute objective
            objective = router.get_length_of_longest_tour()
            return objective

        duration = time.time()
        objective = evaluate()
        duration = time.time() - duration

        log_data = {}
        log_data['gen min obj'] = objective
        log_data['gen mean obj'] = objective
        log_data['gen max obj'] = objective
        log_data['gen time(s)'] = duration
        log_data['total run time(s)'] = duration
        log_data['run best evaluation'] = 1
        log_data['run best time(s)'] = duration
        log_data['run best generation'] = 1
        log_data['run best obj'] = objective
        log_data['run best binary'] = [heuristic_id] * chrom_len
        log_data['run best route'] = router.get_route()
        log_data['run best heuristics'] = [heuristic_id] * chrom_len
        log_data['run time(s)'] =duration

        wandb.config = {
            'instance' : gph.config,
            'routing' : router.config
        }
        
        wandb.init(project="metaga-data", name=gph.name + '-'+ str(seed), config=wandb.config)
        wandb.log(log_data)
        wandb.finish()

if __name__ == '__main__':
    main()