import argparse
import sys
import numpy
import pandas
import re
import csv

from collections import defaultdict

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
            key['run best eval']=[]

        key['run best obj'].append(row['run best obj'])
        key['run best gen'].append(row['run best generation'])
        key['run best eval'].append(row['run best evaluation'])

    return ag_data

def exclude_unbalanced_runs(grouped_data):
    # Initialize variables
    to_remove = []

    # Loop through the groups and find the minimum number of 'ga.random_seed' values
    for num_tours, tour_data in grouped_data.items():
        for instance, instance_data in tour_data.items():
            sum_heuristic_len = 0
            for heuristic_group, heuristic_data in instance_data.items():
                sum_heuristic_len += len(heuristic_data)
                if (len(heuristic_data) != 30):
                    to_remove.append((num_tours, instance, heuristic_group))
                    break

    # Remove unbalanced instances from the grouped data
    for group in to_remove:
        print("Excluding routing.num_tours=" + str(group[0]) + ', instance.name='+group[1] + ' due to unbalanced number of runs on heuristic ' + group[2])
        del grouped_data[group[0]][group[1]]

    return grouped_data    

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


def remove_numeric_suffix(string):
    # Use regular expressions to match the numeric suffix at the end of the string
    match = re.search(r'\d+$', string)
    if match:
        # If a match is found, remove the numeric suffix and return the result
        return string[:match.start()]
    else:
        # If no match is found, return the original string
        return string

def write_statistics_overall(data, file_path):
    temp_data = {}
    for num_tours, tour_data in data.items():
        for instance, instance_data in tour_data.items():
            for heuristic_group, heuristic_data in instance_data.items():
                if not (heuristic_group in temp_data):
                    temp_data[heuristic_group] = {}
                for seed, seed_data in heuristic_data.items():
                    for data, run_data in seed_data.items():
                        if not (data in temp_data[heuristic_group]):
                            temp_data[heuristic_group][data] = []
                        temp_data[heuristic_group][data].append(run_data[0])

    with open(file_path, 'a') as file:
        file.write('overall,avg rr,avg mmmr,avg perc impr,best rr,best mmmr, best perc impr, mann, paired t-test, two-sample t-test\n')
        for data, run_data in temp_data['RR'].items():
            if data != 'run best eval':
                average_rr = numpy.mean(temp_data['RR'][data])
                average_mmmr = numpy.mean(temp_data['MMMR'][data])
                avg_percent_improvement = ((average_rr - average_mmmr) / average_mmmr)
                best_rr = numpy.min(temp_data['RR'][data])
                best_mmmr = numpy.min(temp_data['MMMR'][data])
                best_percent_improvement = ((best_rr - best_mmmr) / best_mmmr)
                mannwhitneyu_test = mannwhitneyu(temp_data['RR'][data], temp_data['MMMR'][data])
                paired_t_test = ttest_rel(temp_data['RR'][data], temp_data['MMMR'][data])
                two_sample_t_test = ttest_ind(temp_data['RR'][data], temp_data['MMMR'][data])
                file.write(data+','+
                        str(average_rr)+','+str(average_mmmr)+','+str(avg_percent_improvement)+','+
                        str(best_rr)+','+str(best_mmmr)+','+str(best_percent_improvement)+','+
                        str(mannwhitneyu_test[1])+','+str(paired_t_test[1])+','+str(two_sample_t_test[1])+'\n')

def write_per_kvalue_statistics(grouped_data, filename):
    results = defaultdict(dict)
    with open(filename, 'a') as f:
        for num_tours, tour_data in grouped_data.items():
            for instance, instance_data in tour_data.items():
                for heuristic_group, heuristic_data in instance_data.items():
                    if not (heuristic_group in results[num_tours]):
                        results[num_tours][heuristic_group] = {}
                    for seed, seed_data in heuristic_data.items():
                        for data, run_data in seed_data.items():
                            if not (data in results[num_tours][heuristic_group]):
                                results[num_tours][heuristic_group][data] = []
                            results[num_tours][heuristic_group][data].append(run_data[0])

            f.write('k='+str(num_tours)+',avg rr,avg mmmr,avg perc impr,best rr,best mmmr, best perc impr, mann, paired t-test, two-sample t-test\n')
            for data, run_data in results[num_tours]['RR'].items():
                if data != 'run best eval':
                    average_rr = numpy.mean(results[num_tours]['RR'][data])
                    average_mmmr = numpy.mean(results[num_tours]['MMMR'][data])
                    avg_percent_improvement = ((average_rr - average_mmmr) / average_mmmr)
                    best_rr = numpy.min(results[num_tours]['RR'][data])
                    best_mmmr = numpy.min(results[num_tours]['MMMR'][data])
                    best_percent_improvement = ((best_rr - best_mmmr) / best_mmmr)
                    mannwhitneyu_test = mannwhitneyu(results[num_tours]['RR'][data], results[num_tours]['MMMR'][data])
                    paired_t_test = ttest_rel(results[num_tours]['RR'][data], results[num_tours]['MMMR'][data])
                    two_sample_t_test = ttest_ind(results[num_tours]['RR'][data], results[num_tours]['MMMR'][data])
                    f.write(data+','+
                            str(average_rr)+','+str(average_mmmr)+','+str(avg_percent_improvement)+','+
                            str(best_rr)+','+str(best_mmmr)+','+str(best_percent_improvement)+','+
                            str(mannwhitneyu_test[1])+','+str(paired_t_test[1])+','+str(two_sample_t_test[1])+'\n')

def write_per_group_statistics(grouped_data, filename):
    results = defaultdict(dict)
    with open(filename, 'a') as f:
        f.write('## Per group\n')
        f.write('Comparing heuristic group RR vs MMMR on individual instance groups, all k-values and all runs:\n')
        for num_tours, tour_data in grouped_data.items():
            for instance, instance_data in tour_data.items():
                group_name = re.sub(r'\d+$', '', instance)
                for heuristic_group, heuristic_data in instance_data.items():
                    if not (group_name in results):
                        results[group_name] = defaultdict(dict)
                    if not (heuristic_group in results[group_name]):
                        results[group_name][heuristic_group] = defaultdict(list)
                    for seed, seed_data in heuristic_data.items():
                        for data, run_data in seed_data.items():
                            results[group_name][heuristic_group][data].append(run_data[0])

        for group_name, group_data in results.items():
            f.write('instance_group='+group_name+',avg rr,avg mmmr,avg perc impr,best rr,best mmmr, best perc impr, mann, paired t-test, two-sample t-test\n')
            for data, run_data in group_data['RR'].items():
                if data != 'run best eval':
                    average_rr = numpy.mean(group_data['RR'][data])
                    average_mmmr = numpy.mean(group_data['MMMR'][data])
                    avg_percent_improvement = ((average_rr - average_mmmr) / average_mmmr)
                    best_rr = numpy.min(group_data['RR'][data])
                    best_mmmr = numpy.min(group_data['MMMR'][data])
                    best_percent_improvement = ((best_rr - best_mmmr) / best_mmmr)
                    mannwhitneyu_test = mannwhitneyu(group_data['RR'][data], group_data['MMMR'][data])
                    paired_t_test = ttest_rel(group_data['RR'][data], group_data['MMMR'][data])
                    two_sample_t_test = ttest_ind(group_data['RR'][data], group_data['MMMR'][data])
                    f.write(data+','+
                            str(average_rr)+','+str(average_mmmr)+','+str(avg_percent_improvement)+','+
                            str(best_rr)+','+str(best_mmmr)+','+str(best_percent_improvement)+','+
                            str(mannwhitneyu_test[1])+','+str(paired_t_test[1])+','+str(two_sample_t_test[1])+'\n')

def write_per_instance_statistics_to_file(grouped_data, filename):
    with open(filename, 'a') as f:
        results = defaultdict(dict)
        f.write('## Per Instance Statistics\n')
        f.write('Comparing heuristic group RR vs MMMR on individual instances, all k-values and all runs:\n')
        for num_tours, tour_data in grouped_data.items():
            for instance, instance_data in tour_data.items():
                for heuristic_group, heuristic_data in instance_data.items():
                    if not (heuristic_group in results[instance]):
                        results[instance][heuristic_group] = {}
                    for seed, seed_data in heuristic_data.items():
                        for data, run_data in seed_data.items():
                            if not (data in results[instance][heuristic_group]):
                                results[instance][heuristic_group][data] = []
                            results[instance][heuristic_group][data].append(run_data[0])

                f.write(' - Instance ' + instance + ':\n')
                temp_results = {}
                for data, run_data in results[instance]['RR'].items():
                    temp_results[data] = {}
                    temp_results[data]['mannwhitneyu'] = mannwhitneyu(results[instance]['RR'][data], results[instance]['MMMR'][data])
                    temp_results[data]['paired_t-test'] = ttest_rel(results[instance]['RR'][data], results[instance]['MMMR'][data])
                    temp_results[data]['two_sample_t-test'] = ttest_ind(results[instance]['RR'][data], results[instance]['MMMR'][data])

                alpha = 0.05
                for data, run_data in temp_results.items():
                    f.write("     - " + data + ':\n')
                    for statistic, statistic_data in run_data.items():
                        p_val = statistic_data[1]
                        if p_val < alpha:
                            f.write("         - [X] " + statistic + " test indicates a significant difference (p-value: "+ str(alpha) + '<' + str(round(p_val,3)) + ")\n")
                        else:
                            f.write("         - [ ] " + statistic + " test indicates no significant difference (p-value: "+ str(alpha) + '<' + str(round(p_val,3)) + ")\n")

def main():
    args = parse_args()
    df = pandas.read_csv(args.data_file)

    # drop duplicates based on these columns
    df = df.drop_duplicates(subset=['routing.num_tours', 'instance.name', 'routing.heuristic_group', 'ga.random_seed'])

    # Remove state == killed or state == crashed
    df = df[df['State'] == 'finished']
        
    # Group the data and track if known
    grouped_data = group_data_by_columns(df, 'routing.num_tours', 'instance.name', 'routing.heuristic_group', 'ga.random_seed')
    
    # Remove unbalanced runs
    grouped_data = exclude_unbalanced_runs(grouped_data)

    # Clears the results file
    filename = 'results.csv'
    with open(filename, 'w') as f:
        pass

    write_statistics_overall(grouped_data, filename)

    write_per_kvalue_statistics(grouped_data, filename)
    write_per_group_statistics(grouped_data, filename)
    # write_per_instance_statistics_to_file(grouped_data, filename)


    # wandb.init(project="metaga-summary", name='best-obj-summary')
    # table = wandb.Table(data=data_best_obj, columns=columns)
    # wandb.log({"avg best objectives": table})

    pass

if __name__ == '__main__':
    main()