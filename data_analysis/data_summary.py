import argparse
import sys
import numpy
import pandas

from scipy.stats import mannwhitneyu

import wandb

from os import path

def parse_args():
    # capture the args with the parser
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-f', '--file', dest='data_file', type=str, required=True, help='filepath to data')
    args = parser.parse_args()

    # check and adjust the parsed args
    args.data_file = args.data_file.replace(' ', '')
    if not path.exists(args.data_file):
        sys.exit("Cannot find instance: " + args.data_file)
    return args

def main():
    args = parse_args()
    df = pandas.read_csv(args.data_file)

    ag_data = {} #[num tours][instance][heuristic]
    # Loop through the rows of the dataframe
    for index, row in df.iterrows():
        # Access data for each column by column name
        num_tours = row['routing.num_tours']
        if not (num_tours in ag_data):
            ag_data[num_tours] = {}

        instance = row['instance.name']
        if not (instance in ag_data[num_tours]):
            ag_data[num_tours][instance] = {}
        
        heuristic_group = row['routing.heuristic_group']
        if not (heuristic_group in ag_data[num_tours][instance]):
            ag_data[num_tours][instance][heuristic_group] = {}

        if not ('run best obj' in ag_data[num_tours][instance][heuristic_group]):
            ag_data[num_tours][instance][heuristic_group]['run best obj']=[]
            ag_data[num_tours][instance][heuristic_group]['run best gen']=[]

        ag_data[num_tours][instance][heuristic_group]['run best obj'].append(row['run best obj'])
        ag_data[num_tours][instance][heuristic_group]['run best gen'].append(row['run best generation'])
    
    data_best_obj = []
    for num_tours in ag_data:
        for instance in ag_data[num_tours]:
            for heuristic_group in ag_data[num_tours][instance]:
                ag_data[num_tours][instance][heuristic_group]['run best obj'] = numpy.array(ag_data[num_tours][instance][heuristic_group]['run best obj'])
                ag_data[num_tours][instance][heuristic_group]['run avg best obj'] = numpy.mean(ag_data[num_tours][instance][heuristic_group]['run best obj'])
                ag_data[num_tours][instance][heuristic_group]['run best gen'] = numpy.array(ag_data[num_tours][instance][heuristic_group]['run best gen'])
                ag_data[num_tours][instance][heuristic_group]['run avg best gen'] = numpy.mean(ag_data[num_tours][instance][heuristic_group]['run best gen'])

            temp_data = [num_tours, instance]
            temp_data.append(round(ag_data[num_tours][instance]['RR']['run avg best obj'], 4))
            temp_data.append(round(ag_data[num_tours][instance]['MMMR']['run avg best obj'], 4))
            data_1 = ag_data[num_tours][instance]['RR']['run best obj']
            data_2 = ag_data[num_tours][instance]['MMMR']['run best obj']
            stat, p_value = mannwhitneyu(data_1, data_2)
            temp_data.append(str(round(p_value, 3)))
            temp_data.append(str(p_value < 0.05))

            temp_data.append(round(ag_data[num_tours][instance]['RR']['run avg best gen'], 4))
            temp_data.append(round(ag_data[num_tours][instance]['MMMR']['run avg best gen'], 4))
            data_1 = ag_data[num_tours][instance]['RR']['run best gen']
            data_2 = ag_data[num_tours][instance]['MMMR']['run best gen']
            stat, p_value = mannwhitneyu(data_1, data_2)
            temp_data.append(str(round(p_value, 3)))
            temp_data.append(str(p_value < 0.05))
            data_best_obj.append(temp_data)
            
    columns=["k", "instance", "RR ave best obj", "MMMR ave best obj", 'obj p-value', 'obj sig_diff', 'RR ave best gen', 'MMMR ave best gen', 'gen p-value', 'gen sig_diff']

    # data_1 = numpy.array(data_best_obj)[:,2].astype(numpy.float32).tolist()
    # data_2 = numpy.array(data_best_obj)[:,3].astype(numpy.float32).tolist()

    # stat, p_value = mannwhitneyu(data_1, data_2)


    wandb.init(project="pygad-tests", name='best-obj-summary')
    table = wandb.Table(data=data_best_obj, columns=columns)
    wandb.log({"avg best objectives": table})

    pass

if __name__ == '__main__':
    main()