import argparse
import sys
import numpy
import pandas

from scipy.stats import mannwhitneyu
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind

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

def group_data_by_columns(dataframe, *column_keywords):
    ag_data = {} #[num tours][instance][heuristic]

    # Loop through the rows of the dataframe
    for index, row in dataframe.iterrows():
        # Access data for each column by column name
        key = ag_data
        for keyword in column_keywords:
            value = row[keyword]
            if value not in key:
                key[value] = {}
            key = key[value]

        if not ('run best obj' in key):
            key['run best obj']=[]
            key['run best gen']=[]

        key['run best obj'].append(row['run best obj'])
        key['run best gen'].append(row['run best generation'])

    return ag_data

def produce_statisctics_for_k_values(grouped_data):
    results = {}
    for num_tours in grouped_data:
        data_1 = grouped_data[num_tours]['RR']['run best obj']
        data_2 = grouped_data[num_tours]['MMMR']['run best obj']
        results[num_tours] = {}
        results[num_tours]['mannwhitneyu'] = mannwhitneyu(data_1, data_2)
        results[num_tours]['paired_t-test'] = ttest_rel(data_1, data_2)
        results[num_tours]['two_sample_t-test'] = ttest_ind(data_1, data_2)

    for num_tours in results:
        alpha = 0.05
        for key, value in results[num_tours].items():
            p_val = value[1]
            if p_val < alpha:
                print("For k = " + str(num_tours) + " the " + key + " test indicates a significant difference (p-value = " + str(p_val) + ")")
            else:
                print("For k = " + str(num_tours) + " the " + key + " test indicates no significant difference (p-value = " + str(p_val) + ")")
    print()
    return results

def main():
    args = parse_args()
    df = pandas.read_csv(args.data_file)

    grouped_data = group_data_by_columns(df, 'routing.num_tours', 'routing.heuristic_group')
    produce_statisctics_for_k_values(grouped_data)

    # Compute the per instance statistics
    grouped_data = group_data_by_columns(df, 'routing.num_tours', 'instance.name', 'routing.heuristic_group')

    data_best_obj = []
    for num_tours in grouped_data:
        for instance in grouped_data[num_tours]:
            for heuristic_group in grouped_data[num_tours][instance]:
                grouped_data[num_tours][instance][heuristic_group]['run best obj'] = numpy.array(grouped_data[num_tours][instance][heuristic_group]['run best obj'])
                grouped_data[num_tours][instance][heuristic_group]['run avg best obj'] = numpy.mean(grouped_data[num_tours][instance][heuristic_group]['run best obj'])
                grouped_data[num_tours][instance][heuristic_group]['run best gen'] = numpy.array(grouped_data[num_tours][instance][heuristic_group]['run best gen'])
                grouped_data[num_tours][instance][heuristic_group]['run avg best gen'] = numpy.mean(grouped_data[num_tours][instance][heuristic_group]['run best gen'])

            # add means and compute statistics
            temp_data = [num_tours, instance]
            temp_data.append(round(grouped_data[num_tours][instance]['RR']['run avg best obj'], 4))
            temp_data.append(round(grouped_data[num_tours][instance]['MMMR']['run avg best obj'], 4))
            data_1 = grouped_data[num_tours][instance]['RR']['run best obj']
            data_2 = grouped_data[num_tours][instance]['MMMR']['run best obj']
            stat, p_value = mannwhitneyu(data_1, data_2)
            temp_data.append(str(round(p_value, 3)))
            temp_data.append(str(p_value < 0.05))

            temp_data.append(round(grouped_data[num_tours][instance]['RR']['run avg best gen'], 4))
            temp_data.append(round(grouped_data[num_tours][instance]['MMMR']['run avg best gen'], 4))
            data_1 = grouped_data[num_tours][instance]['RR']['run best gen']
            data_2 = grouped_data[num_tours][instance]['MMMR']['run best gen']
            stat, p_value = mannwhitneyu(data_1, data_2)
            temp_data.append(str(round(p_value, 3)))
            temp_data.append(str(p_value < 0.05))
            data_best_obj.append(temp_data)
            
    columns=["k", "instance", "RR ave best obj", "MMMR ave best obj", 'obj p-value', 'obj sig_diff', 'RR ave best gen', 'MMMR ave best gen', 'gen p-value', 'gen sig_diff']

    # data_1 = numpy.array(data_best_obj)[:,2].astype(numpy.float32).tolist()
    # data_2 = numpy.array(data_best_obj)[:,3].astype(numpy.float32).tolist()

    # stat, p_value = mannwhitneyu(data_1, data_2)


    wandb.init(project="metaga-summary", name='best-obj-summary')
    table = wandb.Table(data=data_best_obj, columns=columns)
    wandb.log({"avg best objectives": table})

    pass

if __name__ == '__main__':
    main()