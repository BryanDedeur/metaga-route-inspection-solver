import argparse
import sys
import numpy
import pandas
import re
import csv
import random

from collections import defaultdict

from scipy.stats import mannwhitneyu
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import ranksums
from scipy.stats import shapiro

import wandb

from os import path

def get_data_from_wandb_api():
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("bryandedeur/metaga-data")

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pandas.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })

    runs_df.to_csv("export2.csv")

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

def average_data_by_columns(dataframe, *column_keywords):
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

        for group_name, group_data in results.items():
            f.write('instance='+group_name+',avg rr,avg mmmr,avg perc impr,best rr,best mmmr, best perc impr, mann, paired t-test, two-sample t-test\n')
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
                    
def write_per_kvalue_per_instance_objective_percent_improvement(grouped_data, filename):
    with open(filename, 'a') as f:
        order_of_heuristics = ['MMMR', 'RR', 'MIN-MIN', 'MIN-MEDIAN', 'MIN-MAX', 'MIN-RANDOM']
        f.write('## Per K-Value, Per Instance, Avg Objective Percent Improvement \n')
        f.write('Comparing all heuristic groups on individual instances, all k-values and all runs avg best objective:\n')
        for num_tours, tour_data in grouped_data.items():
            output = 'k='+str(num_tours)+'\n'
            f.write(output+'\n')
            for instance, instance_data in tour_data.items():
                results = defaultdict(dict)
                # output the header
                for heuristic_group, heuristic_data in instance_data.items():
                    results[heuristic_group] = []
                    for seed, seed_data in heuristic_data.items():
                        results[heuristic_group].append(seed_data['run best obj'][0])
            
                output = 'instance='+instance+','
                avg_best_mmmr_obj = numpy.mean(results['MMMR'])
                for heuristic_group in order_of_heuristics:
                    if len(results[heuristic_group]) != 0:
                        if heuristic_group == 'MMMR':
                            output += str(avg_best_mmmr_obj)+','
                        else:
                            avg_best_obj = numpy.mean(results[heuristic_group])
                            perc_improvement = -1 * ((avg_best_obj - avg_best_mmmr_obj) / avg_best_mmmr_obj)
                            output += str(perc_improvement)+','
                    else:
                        output += ','
                f.write(output + '\n') 

def filter_dataframe(df, filter_func):
    """
    Filters a dataframe based on a given lambda function.

    Parameters:
    df (pandas.DataFrame): The input dataframe to filter.
    filter_func (function): A lambda function that takes a row of the dataframe as input and returns a boolean value.

    Returns:
    pandas.DataFrame: A subset dataframe containing only the rows where filter_func returns True.
    """
    mask = df.apply(filter_func, axis=1)
    return df.loc[mask, :]

def group_data(df, keywords):
    grouped_dict = {}

    for index, row in df.iterrows():
        key = grouped_dict
        for keyword in keywords:
            keys = keyword.split('.')
            value = row
            for k in keys:
                value = value.get(k, None)
                if value is None:
                    break
            if value is not None:
                if value not in key:
                    key[value] = {}
                key = key[value]

    return grouped_dict

def write_analyzed_data(dataframe, instance, filename):
    sub_dataframe = filter_dataframe(dataframe, 
        lambda x: 'routing' in x['config'] and (
        instance in x['config']['instance']['name']
    ))

    # Build a grouped dictionary by instance name and depot group
    grouped = {}

    for index, row in sub_dataframe.iterrows():
        instance_name = row['config']['instance']['name']
        depot_group = row['config']['routing']['depot_group']
        if instance_name not in grouped:
            grouped[instance_name] = {}
        if depot_group not in grouped[instance_name]:
            grouped[instance_name][depot_group] = {'best obj' : float('inf'), 'avg best obj' : [], 'avg time till best(s)': []}
        if row['summary']['run best obj'] < grouped[instance_name][depot_group]['best obj']:
            grouped[instance_name][depot_group]['best obj'] = row['summary']['run best obj']
        grouped[instance_name][depot_group]['avg best obj'].append(row['summary']['run best obj'])
        grouped[instance_name][depot_group]['avg time till best(s)'].append(row['summary']['run best time(s)'])

    for key_1, row_1 in grouped.items():
        for key_2, row_2 in row_1.items():
            grouped[key_1][key_2]['avg best obj'] = numpy.min(grouped[key_1][key_2]['avg best obj'])
            grouped[key_1][key_2]['avg time till best(s)'] = numpy.min(grouped[key_1][key_2]['avg time till best(s)'])

    # Build table
    headers = ['Instance', 'Single Depot Best Obj', 'Multi Depot Best Obj', 'Avg Best Obj Improvement', 'Avg Time Till Best Obj Improvement(s)']
    rows = [instance + '1', instance + '2', instance + '3', instance + '4', instance + '5']
    table = []
    table.append(headers)
    for i in range(len(rows)):
        table.append([])
        table[i+1].append(rows[i])
        table[i+1].append(grouped[rows[i]]['single']['best obj'])
        table[i+1].append(grouped[rows[i]]['multi']['best obj'])
        table[i+1].append((grouped[rows[i]]['single']['avg best obj'] - grouped[rows[i]]['multi']['avg best obj']) / grouped[rows[i]]['multi']['avg best obj'])
        table[i+1].append((grouped[rows[i]]['single']['avg time till best(s)'] - grouped[rows[i]]['multi']['avg time till best(s)']) / grouped[rows[i]]['multi']['avg time till best(s)'])

    # Open a new CSV file in write mode
    with open(filename, 'a', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        
        # Write each row of data to the CSV file
        for row in table:
            writer.writerow(row)

def main():
    # get_data_from_wandb_api()

    args = parse_args()

    ahr_df = pandas.read_csv('data_analysis/ahr_data.csv',header=0, index_col=0)
    metaga_df = pandas.read_csv('data_analysis/metaga_data.csv',header=0, index_col=0)
    dega_df = pandas.read_csv('data_analysis/dega_data.csv',header=0, index_col=0)
    max_lb_df = pandas.read_csv('data_analysis/max_lb.csv',header=0, index_col=0)

    # Clears the results file
    filename = 'stat_results.csv'
    with open(filename, 'w') as f:
        pass

    # filename = 'stat_results.csv'
    # with open(filename, 'a') as f:
    #     instances = ['gdb1','gdb2','gdb3','gdb4','gdb5','gdb6','gdb7','gdb8','gdb9','gdb10','gdb11','gdb12','gdb13','gdb14','gdb15','gdb16','gdb17','gdb18','gdb19','gdb20', 'gdb21', 'gdb22', 'gdb23']   
    #     k_values = [2,4,8]
    #     for instance in instances:
    #         f.write(instance + ',')
    #         for k_value in k_values:
    #             for row in metaga_df.iterrows():
    #                 if row[0] == instance and row[1]['k'] == k_value:
    #                     f.write(str(round(row[1]['best'], 3)) + ',')
    #                     break
    #         f.write('\n')

    def percentage_improvement(a, b):
        return (a-b)/b


    # # Generate percentage improvement for METAGA vs DEGA vs AHR on GDB instances ----------------------------------------------------------------------------------------
    # filename = 'stat_results.csv'
    # with open(filename, 'a') as f:
    #     instances = ['gdb1','gdb2','gdb3','gdb4','gdb5','gdb6','gdb7','gdb8','gdb9','gdb10','gdb11','gdb12','gdb13','gdb14','gdb15','gdb16','gdb17','gdb18','gdb19','gdb20', 'gdb21', 'gdb22', 'gdb23']    
    #     k_values = [2,4,8]
    #     perc_improv = {}
    #     perc_improv['META VS AHR'] = [0] * 23
    #     perc_improv['META VS DEGA'] = [0] * 23
    #     perc_improv['META VS MAX-LB'] = [0] * 23

    #     for k_value in k_values:
    #         compare_list = {}
    #         compare_list['DEGA'] = [] 
    #         compare_list['META'] = []
    #         compare_list['AHR'] = []

    #         max_lb_list = []

    #         for instance in instances:
    #             for row in max_lb_df.iterrows():
    #                 if row[0] == instance:
    #                     max_lb_list.append(row[1][k_value])
    #                     break

    #         for i in range(len(instances)):
    #             for row in metaga_df.iterrows():
    #                 if row[0] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['META'].append(row[1]['best'])
    #                     break
    #             for row in dega_df.iterrows():
    #                 if row[1]['Problem Instance'] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['DEGA'].append(row[1]['Best Longest Route'])
    #                     break
    #             for row in ahr_df.iterrows():
    #                 if row[0] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['AHR'].append(row[1]['TInf'])
    #                     break

    #         for i in range(23):
    #             perc_improv['META VS AHR'][i] += -1 * percentage_improvement(compare_list['META'][i], compare_list['AHR'][i])
    #             perc_improv['META VS DEGA'][i] += -1 * percentage_improvement(compare_list['META'][i], compare_list['DEGA'][i])
    #             perc_improv['META VS MAX-LB'][i] += -1 * percentage_improvement(compare_list['META'][i], max_lb_list[i])

    #         f.write('META VS AHR k='+ str(k_value) + ', META VS DEGA k=' + str(k_value) + ', META VS MAX-LB k=' + str(k_value) +'\n')
    #         for i in range(len(compare_list['DEGA'])):
    #             f.write(str(-1 * percentage_improvement(compare_list['META'][i], compare_list['AHR'][i])) + ',' + str(-1 * percentage_improvement(compare_list['META'][i], compare_list['DEGA'][i])) + ',' + str(-1 * percentage_improvement(compare_list['META'][i], max_lb_list[i])) + '\n')

    #     f.write('instance, TS, DEGA, MAX-LB\n')
    #     for i in range(23):
    #         f.write(str(i+1) + ', ' + str(perc_improv['META VS AHR'][i]/3) + ',' + str(perc_improv['META VS DEGA'][i]/3) + ',' + str(perc_improv['META VS MAX-LB'][i]/3)+ '\n')

    # print()

    # # Generate Statistics for METAGA vs DEGA vs AHR on GDB instances ----------------------------------------------------------------------------------------
    # def normalize(value1, value2):
    #     # return (value1 - value2)/value2
    #     return (value1 - value2)

    # filename = 'stat_results.csv'
    # with open(filename, 'a') as f:
    #     instances = ['gdb1','gdb2','gdb3','gdb4','gdb5','gdb6','gdb7','gdb8','gdb9','gdb10','gdb11','gdb12','gdb13','gdb14','gdb15','gdb16','gdb17','gdb18','gdb19','gdb20', 'gdb21', 'gdb22', 'gdb23']    
    #     k_values = [2,4,8]
    #     for k_value in k_values:
    #         compare_list = {}
    #         compare_list['DEGA'] = []
    #         compare_list['META'] = []
    #         compare_list['AHR'] = []

    #         max_lb_list = []

    #         for instance in instances:
    #             for row in max_lb_df.iterrows():
    #                 if row[0] == instance:
    #                     max_lb_list.append(row[1][k_value])
    #                     break

    #         for i in range(len(instances)):
    #             for row in metaga_df.iterrows():
    #                 if row[0] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['META'].append(normalize(row[1]['best'], max_lb_list[i]))
    #                     break
    #             for row in dega_df.iterrows():
    #                 if row[1]['Problem Instance'] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['DEGA'].append(normalize(row[1]['Best Longest Route'], max_lb_list[i]))
    #                     break
    #             for row in ahr_df.iterrows():
    #                 if row[0] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['AHR'].append(normalize(row[1]['TInf'],max_lb_list[i]))
    #                     break
            
    #         # t_statistic, p_value = ttest_ind(compare_list['DEGA'], compare_list['META'])
    #         # f.write(str(round(t_statistic, 3)) + ',' + str(round(p_value, 3))+',')

    #         # print('Independent two-sample t-test: ' + str(p_value) + ' ' +str(p_value > 0.05))
    #         f.write('META k='+ str(k_value) + ', AHR k='+ str(k_value) + ', DEGA k=' + str(k_value) + '\n')
    #         for i in range(len(compare_list['DEGA'])):
    #             f.write(str(compare_list['META'][i]) + ',' + str(compare_list['AHR'][i]) + ',' + str(compare_list['DEGA'][i]) + '\n')

    # print()

    #     # Generate percentage improvement for METAGA vs DEGA on BRIDGE instances ----------------------------------------------------------------------------------------
    # filename = 'stat_results.csv'
    # with open(filename, 'a') as f:
    #     instances = ['howe1','howe2','howe3','howe4','howe5','pratt1','pratt2','pratt3','pratt4','pratt5','ktruss1','ktruss2','ktruss3','ktruss4','ktruss5','warren1','warren2','warren3','warren4','warren5']     
    #     k_values = [2,4,8]
    #     perc_improv = {}
    #     perc_improv['META VS DEGA'] = [0] * 20
    #     perc_improv['META VS MAX-LB'] = [0] * 20

    #     for k_value in k_values:
    #         compare_list = {}
    #         compare_list['DEGA'] = [] 
    #         compare_list['META'] = []

    #         max_lb_list = []

    #         for instance in instances:
    #             for row in max_lb_df.iterrows():
    #                 if row[0] == instance:
    #                     max_lb_list.append(row[1][k_value])
    #                     break

    #         for i in range(len(instances)):
    #             for row in metaga_df.iterrows():
    #                 if row[0] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['META'].append(row[1]['best'])
    #                     break
    #             for row in dega_df.iterrows():
    #                 if row[1]['Problem Instance'] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['DEGA'].append(row[1]['Best Longest Route'])
    #                     break

    #         for i in range(20):
    #             perc_improv['META VS DEGA'][i] += -1 * percentage_improvement(compare_list['META'][i], compare_list['DEGA'][i])
    #             perc_improv['META VS MAX-LB'][i] += -1 * percentage_improvement(compare_list['META'][i], max_lb_list[i])

    #         f.write('META VS DEGA k=' + str(k_value) + ', META VS MAX-LB k=' + str(k_value) +'\n')
    #         for i in range(20):
    #             f.write(str(-1 * percentage_improvement(compare_list['META'][i], compare_list['DEGA'][i])) + ',' + str(-1 * percentage_improvement(compare_list['META'][i], max_lb_list[i])) + '\n')

    #     f.write('DEGA, MAX-LB\n')
    #     for i in range(20):
    #         f.write(str(perc_improv['META VS DEGA'][i]/3) + ',' + str(perc_improv['META VS MAX-LB'][i]/3)+ '\n')

    # print()

    # Generate Statistics for METAGA vs DEGA on BRIDGE instances ----------------------------------------------------------------------------------------
    # def normalize(value1, value2):
    #     return (value1 - value2)/value2
    #     # return (value1 - value2)

    # filename = 'stat_results.csv'
    # with open(filename, 'a') as f:
    #     instances = ['howe1','howe2','howe3','howe4','howe5','pratt1','pratt2','pratt3','pratt4','pratt5','ktruss1','ktruss2','ktruss3','ktruss4','ktruss5','warren1','warren2','warren3','warren4','warren5']     
    #     k_values = [2,4,8]
    #     for k_value in k_values:
    #         compare_list = {}
    #         compare_list['DEGA'] = []
    #         compare_list['META'] = []

    #         max_lb_list = []

    #         for instance in instances:
    #             for row in max_lb_df.iterrows():
    #                 if row[0] == instance:
    #                     max_lb_list.append(row[1][k_value])
    #                     break

    #         for i in range(len(instances)):
    #             for row in metaga_df.iterrows():
    #                 if row[0] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['META'].append(normalize(row[1]['best'], max_lb_list[i]))
    #                     break
    #             for row in dega_df.iterrows():
    #                 if row[1]['Problem Instance'] == instances[i] and row[1]['k'] == k_value:
    #                     compare_list['DEGA'].append(normalize(row[1]['Best Longest Route'], max_lb_list[i]))
    #                     break

    #         f.write('META k='+ str(k_value) + ', DEGA k=' + str(k_value) + '\n')
    #         for i in range(len(compare_list['DEGA'])):
    #             f.write(str(compare_list['META'][i]) + ',' + str(compare_list['DEGA'][i]) + '\n')

    # print()

    # # Define a lambda function to extract the run data, configuration data, and run name
    # extract_data = lambda x: (eval(x['summary']), eval(x['config']), x['name'])

    # # Apply the lambda function to each row in the dataframe and store the results in separate columns
    # mass_df[['summary', 'config', 'name']] = mass_df.apply(extract_data, axis=1, result_type='expand')

    # # drop duplicates based on these columns
    # df = df.drop_duplicates(subset=['routing.num_tours', 'instance.name', 'routing.heuristic_group', 'ga.random_seed'])


    # Remove state == killed or state == crashed
    # df = df[df['State'] == 'finished']
        
    # Group the data and track if known
    # grouped_data = group_data_by_columns(df, 'routing.depot_group', 'routing.num_tours', 'instance.name', 'routing.heuristic_group', 'ga.random_seed')

    # # Add avg best obj and avg best time to grouped data
    # grouped_data = add_avg_best_obj_and_avg_time_to_reach_best(grouped_data)

    # Clears the results file
    filename = 'results.csv'
    with open(filename, 'w') as f:
        pass

    # --------------------------------------------------------------------
    # Section 2: 1 Depot vs Multi Depot
    # --------------------------------------------------------------------

    # subset_df = filter_dataframe(df, lambda x: 'routing' in x['config'] and (
    #     x['config']['routing'].get('depot_group') == 'single' or 
    #     x['config']['routing'].get('depot_group') == 'multi'
    # ))
    


    # --------------------------------------------------------------------
    # Subsection 2.1: MetaGA 1 Depot vs MetaGA Multi Depot
    # --------------------------------------------------------------------
    mass_df = pandas.read_csv('data_analysis/data_export.csv', header=0, index_col=0)

    # Define a lambda function to extract the run data, configuration data, and run name
    extract_data = lambda x: (eval(x['summary']), eval(x['config']), x['name'])

    # Apply the lambda function to each row in the dataframe and store the results in separate columns
    mass_df[['summary', 'config', 'name']] = mass_df.apply(extract_data, axis=1, result_type='expand')

    subset_df = filter_dataframe(mass_df, lambda x: 'routing' in x['config'] and (
        x['config']['routing'].get('heuristic_group') == 'MMMR'
    ))

    depot_groups = ['single', 'multi']

    sub_subset_dfs = {}
    for depot_group in depot_groups:
        sub_subset_dfs[depot_group] = filter_dataframe(subset_df, lambda x: 'routing' in x['config'] and (
            x['config']['routing'].get('depot_group') == depot_group
        ))
        

    # # Compare metaga with ahr
    # filename = 'stat_results.csv'
    # with open(filename, 'a') as f:
    #     instances = ['howe1','howe2','howe3','howe4','howe5','pratt1','pratt2','pratt3','pratt4','pratt5','ktruss1','ktruss2','ktruss3','ktruss4','ktruss5','warren1','warren2','warren3','warren4','warren5']
    #     col_data = {}
    #     col_data[2] = []
    #     col_data['2stat'] = []
    #     col_data[4] = []
    #     col_data['4stat'] = []
    #     col_data[8] = []
    #     col_data['8stat'] = []
    #     col_data['avg'] = []

    #     row_data = []
    #     for instance in instances:
    #         f.write(instance + ' & ')
    #         k_values = [2,4,8]
    #         for k_value in k_values:
    #             compare_list = {}
    #             this_method = depot_groups[0]
    #             other_method = depot_groups[1]
    #             compare_list[this_method] = []
    #             compare_list[other_method] = []
    #             for row in sub_subset_dfs[this_method].iterrows():
    #                 # METHOD 1
    #                 if row[1]['config']['instance']['name'] == instance and row[1]['config']['routing']['num_tours'] == k_value:
    #                     compare_list[this_method].append(row[1]['summary']['run best obj'])
    #             for row in sub_subset_dfs[other_method].iterrows():
    #                 # METHOD 2
    #                 if row[1]['config']['instance']['name'] == instance and row[1]['config']['routing']['num_tours'] == k_value:
    #                     compare_list[other_method].append(row[1]['summary']['run best obj'])
            
    #             # # avg obj
    #             # f.write( + ',')

    #             # # best obj method 1
    #             # f.write( + ',')

    #             # # best obj method 2
    #             # f.write( + ',')
    #             num_runs = len(compare_list[this_method])
    #             if num_runs != 30:
    #                 print("WARN: " + this_method + ' ' + instance + ' ' + str(k_value) + ' runs size is: ' + num_runs)
    #             num_runs = len(compare_list[other_method])
    #             if num_runs != 30:
    #                 print("WARN: " + other_method + ' ' + instance + ' ' + str(k_value) + ' runs size is: ' + num_runs)
                
    #             # percentage improvement
    #             # if negative single has a lower objective value, if positive single has a larger objective value
    #             avg1 = numpy.average(compare_list[this_method])
    #             avg2 = numpy.average(compare_list[other_method])
    #             pi = round(100*(avg1-avg2)/avg2,2)
    #             row_data.append(pi)
    #             col_data[k_value].append(pi)
    #             f.write(str(pi) + '\% & ')

    #             # t_statistic, p_value = mannwhitneyu(compare_list[this_method], compare_list[other_method])
    #             t_statistic, p_value = ttest_ind(compare_list[other_method], compare_list[this_method], equal_var=False, alternative='less')

    #             col_data[str(k_value) + "stat"].append(p_value)

    #             # out = False
    #             # if  < 0.05:
    #             #     out = True
    #             out = ''
    #             if p_value > 0.01:
    #                 out = str(round(p_value, 2))
    #             else:
    #                 out = str(format(p_value, ".1e"))
    #             if p_value > 0.05:
    #                 out = '{\\bf ' + out + '}'
    #             f.write(out +' & ')

    #         avg = round(numpy.average(row_data),2)
    #         col_data['avg'].append(avg)
    #         f.write(str(avg) + '\%' + '\n')
    #         row_data = []

    #     for key in col_data.keys():
    #         if isinstance(key, str):
    #             if key == 'avg':
    #                 f.write('{\\bf ' + str(round(numpy.average(col_data[key]),2)) +'\%}')
    #             else:
    #                 f.write('{\\bf ' +format(numpy.average(col_data[key]),".3")+'} & ')
    #         else:
    #             f.write('{\\bf ' +str(round(numpy.average(col_data[key]),2)) +'\%} & ')
    #     f.write('\n')
    # print(0)

    # heuristic_groups = ['META', 'DEGA']

    # data = {}
    # data[heuristic_groups[0]] = pandas.read_csv('data_analysis/meta-multi-depot-data.csv',header=0, index_col=0)
    # data[heuristic_groups[1]] = pandas.read_csv('data_analysis/dega-multi-depot-data.csv',header=0, index_col=0)

    # # Compare metaga with ahr
    # filename = 'stat_results.csv'
    # with open(filename, 'a') as f:
    #     instances = ['howe1','howe2','howe3','howe4','howe5','pratt1','pratt2','pratt3','pratt4','pratt5','ktruss1','ktruss2','ktruss3','ktruss4','ktruss5','warren1','warren2','warren3','warren4','warren5']
    #     col_data = {}
    #     col_data[2] = []
    #     col_data['2stat'] = []
    #     col_data[4] = []
    #     col_data['4stat'] = []
    #     col_data[8] = []
    #     col_data['8stat'] = []
    #     col_data['avg'] = []

    #     row_data = []
    #     for instance in instances:
    #         f.write(instance + ' & ')
    #         k_values = [2,4,8]
    #         for k_value in k_values:
    #             compare_list = {}
    #             this_method = heuristic_groups[0]
    #             other_method = heuristic_groups[1]
    #             compare_list[this_method] = []
    #             compare_list[other_method] = []
    #             for row in data[this_method].iterrows():
    #                 # METHOD 1
    #                 if row[0] == instance and row[1]['k'] == k_value:
    #                     compare_list[this_method].append(row[1]['Objective'])
    #             for row in data[other_method].iterrows():
    #                 # METHOD 2
    #                 if row[0] == instance and row[1]['k'] == k_value:
    #                     compare_list[other_method].append(row[1]['Objective'])
            
    #             num_runs = len(compare_list[this_method])
    #             if num_runs != 30:
    #                 print("WARN: " + this_method + ' ' + instance + ' ' + str(k_value) + ' runs size is: ' + num_runs)
    #             num_runs = len(compare_list[other_method])
    #             if num_runs != 30:
    #                 print("WARN: " + other_method + ' ' + instance + ' ' + str(k_value) + ' runs size is: ' + num_runs)
                
    #             # percentage improvement
    #             # if negative single has a lower objective value, if positive single has a larger objective value
    #             avg1 = numpy.average(compare_list[this_method])
    #             avg2 = numpy.average(compare_list[other_method])
    #             pi = round((100 * ((avg2-avg1)/avg1)),2)
    #             row_data.append(pi)
    #             col_data[k_value].append(pi)
    #             out = ''
    #             if pi > 50:
    #                 f.write('{\\bf' + str(pi) + '\%} & ')
    #             else:
    #                 f.write(str(pi) + '\% & ')

    #             t_statistic, p_value = ttest_ind(compare_list[other_method], compare_list[this_method], equal_var=False)

    #             col_data[str(k_value) + "stat"].append(p_value)

    #             out = ''
    #             if p_value > 0.01:
    #                 out = str(round(p_value, 2))
    #             else:
    #                 out = str(format(p_value, ".1e"))
    #             if p_value > 0.05:
    #                 out = '{\\bf ' + out + '}'
    #             f.write(out +' & ')

    #         avg = round(numpy.average(row_data),2)
    #         col_data['avg'].append(avg)
    #         f.write(str(avg) + '\%' + '\n')
    #         row_data = []

    #     for key in col_data.keys():
    #         if isinstance(key, str):
    #             if key == 'avg':
    #                 f.write('{\\bf ' + str(round(numpy.average(col_data[key]),2)) +'\%}')
    #             else:
    #                 f.write('{\\bf ' +format(numpy.average(col_data[key]),".3")+'} & ')
    #         else:
    #             f.write('{\\bf ' +str(round(numpy.average(col_data[key]),2)) +'\%} & ')
    #     f.write('\n')
    # print(0)


    heuristic_groups = ['META', 'CPP']

    data = {}
    data[heuristic_groups[0]] = pandas.read_csv('data_analysis/meta-multi-depot-data.csv',header=0, index_col=0)
    data[heuristic_groups[1]] = pandas.read_csv('data_analysis/max_lb.csv',header=0, index_col=0)

    # Compare metaga with ahr
    filename = 'stat_results.csv'
    with open(filename, 'a') as f:
        instances = ['howe1','howe2','howe3','howe4','howe5','pratt1','pratt2','pratt3','pratt4','pratt5','ktruss1','ktruss2','ktruss3','ktruss4','ktruss5','warren1','warren2','warren3','warren4','warren5']
        col_data = {}
        col_data[2] = []
        col_data['2stat'] = []
        col_data[4] = []
        col_data['4stat'] = []
        col_data[8] = []
        col_data['8stat'] = []
        col_data['avg'] = []

        row_data = []
        for instance in instances:
            f.write(instance + ' & ')
            k_values = [2,4,8]
            for k_value in k_values:
                compare_list = {}
                this_method = heuristic_groups[0]
                other_method = heuristic_groups[1]
                compare_list[this_method] = []
                compare_list[other_method] = []
                for row in data[this_method].iterrows():
                    # METHOD 1
                    if row[0] == instance and row[1]['k'] == k_value:
                        compare_list[this_method].append(row[1]['Objective'])
                for row in data[other_method].iterrows():
                    # METHOD 2
                    if row[0] == instance:
                        compare_list[other_method].append(row[1][1]/k_value)
            
                num_runs = len(compare_list[this_method])
                if num_runs != 30:
                    print("WARN: " + this_method + ' ' + instance + ' ' + str(k_value) + ' runs size is: ' + num_runs)
                # num_runs = len(compare_list[other_method])
                # if num_runs != 30:
                #     print("WARN: " + other_method + ' ' + instance + ' ' + str(k_value) + ' runs size is: ' + num_runs)
                
                # percentage improvement
                # if negative single has a lower objective value, if positive single has a larger objective value
                avg1 = numpy.min(compare_list[this_method])
                avg2 = numpy.average(compare_list[other_method])
                percentage_improvement = round(100*(avg2-avg1)/avg1,2)
                row_data.append(percentage_improvement)
                col_data[k_value].append(percentage_improvement)
                f.write(str(percentage_improvement) + '\% & ')

                # t_statistic, p_value = mannwhitneyu(compare_list[this_method], compare_list[other_method])
                temp = [avg2] * 30
                t_statistic, p_value = ranksums(compare_list[this_method], [avg2])

                col_data[str(k_value) + "stat"].append(p_value)

                # out = False
                # if  < 0.05:
                #     out = True
                f.write(str(format(p_value, ".2e")) +' & ')

            avg = round(numpy.average(row_data),2)
            col_data['avg'].append(avg)
            f.write(str(avg) +'\% \n')
            row_data = []

        for key in col_data.keys():
            if isinstance(key, str):
                if key == 'avg':
                    f.write('{\\bf ' + str(round(numpy.average(col_data[key]),2)) +'\%}')
                else:
                    f.write('{\\bf ' + format(numpy.average(col_data[key]),".3")+'} & ')
            else:
                f.write('{\\bf ' +str(round(numpy.average(col_data[key]),2)) +'\%} & ')
        f.write('\n')
        for key in col_data.keys():
            if isinstance(key, str):
                if key == 'avg':
                    f.write('{\\bf ' + str(round(numpy.var(col_data[key]),2)) +'\%}')
                else:
                    f.write('{\\bf ' + format(numpy.var(col_data[key]),".3")+'} & ')
            else:
                f.write('{\\bf ' +str(round(numpy.var(col_data[key]),2)) +'\%} & ')
        f.write('\n')
        for key in col_data.keys():
            if isinstance(key, str):
                if key == 'avg':
                    f.write('{\\bf ' + str(round(numpy.std(col_data[key]),2)) +'\%}')
                else:
                    f.write('{\\bf ' + format(numpy.std(col_data[key]),".3")+'} & ')
            else:
                f.write('{\\bf ' +str(round(numpy.std(col_data[key]),2)) +'\%} & ')
        f.write('\n')

    print(0)



    # Clears the results file
    # filename = 'results.csv'
    # with open(filename, 'a') as f:
    #     instance_names = ['gdb1','gdb2','gdb3','gdb4','gdb5','gdb6','gdb7','gdb8','gdb9','gdb10','gdb11','gdb12','gdb13','gdb14','gdb15','gdb16','gdb17','gdb18','gdb19','gdb20', 'gdb21', 'gdb22', 'gdb23']
    #     for instance in instance_names:
    #         f.write(instance + ',')
    #         for k_value in k_values:
    #             compare_set_df = {}
    #             for h_group in heuristic_groups:
    #                 temp = filter_dataframe(sub_subset_dfs[h_group], lambda x: 'routing' in x['config'] and (
    #                     x['config']['routing']['num_tours'] == k_value and
    #                     x['config']['instance']['name'] == instance
    #                 ))
    #                 compare_set_df[h_group] = []
    #                 for run in temp.values:
    #                     compare_set_df[h_group].append(run[0]['run best obj'])
    #             if len(compare_set_df[heuristic_groups[0]]) != len(compare_set_df[heuristic_groups[1]]):
    #                 print("Cannot compare: " + "k=" + str(k_value) + " instance=" + instance)
    #                 f.write('NA,NA,')
    #                 continue
                
    #             t_statistic, p_value = mannwhitneyu(compare_set_df[heuristic_groups[0]], compare_set_df[heuristic_groups[1]])
    #             sig_diff = p_value > 0.05
    #             print("k=" + str(k_value) + " instance=" + instance + " sig_diff=" + str(sig_diff))
    #             f.write(str(p_value) + ',' +str(sig_diff) +',')
    #         f.write('\n')

    print()

    # instance_k2_metaga_sub_subset_df = filter_dataframe(k2_metaga_sub_subset_df, lambda x: 'routing' in x['config'] and (
    #     x['config']['instance']['name'] == "gdb1"
    # ))

    # --------------------------------------------------------------------
    # Subsubsection 2.1.1: k=2
    # --------------------------------------------------------------------

    # k2_metaga_single_metaga_multi_df = filter_dataframe(metaga_single_metaga_multi_df, lambda x: 'routing' in x['config'] and (
    #     x['config']['routing']['num_tours'] == 2
    # ))

    # # pratt
    # write_analyzed_data(k2_metaga_single_metaga_multi_df, 'pratt', filename)

    # # howe
    # write_analyzed_data(k2_metaga_single_metaga_multi_df, 'howe', filename)

    # # warren
    # write_analyzed_data(k2_metaga_single_metaga_multi_df, 'warren', filename)

    # # ktruss
    # write_analyzed_data(k2_metaga_single_metaga_multi_df, 'ktruss', filename)

    # Subsubsection 2.1.1: k=4

    # pratt

    # howe

    # warren

    # ktruss

    # Subsubsection 2.1.1: k=8

    # pratt

    # howe

    # warren

    # ktruss

    # Subsection 2.2: MetaGA Multi Depot vs DEGA Multi Depot


    # # For comparing all heuristic groups to eachother without statistical tests
    # write_per_kvalue_per_instance_objective_percent_improvement(grouped_data, filename)
    
    # # Remove unbalanced runs
    # grouped_data = exclude_unbalanced_runs(grouped_data)

    # For comparing one heuristic to another with statistical tests
    # write_statistics_overall(grouped_data, filename)
    # write_per_kvalue_statistics(grouped_data, filename)
    # write_per_group_statistics(grouped_data, filename)
    # write_per_instance_statistics_to_file(grouped_data, filename)




    # wandb.init(project="metaga-summary", name='best-obj-summary')
    # table = wandb.Table(data=data_best_obj, columns=columns)
    # wandb.log({"avg best objectives": table})

    pass

if __name__ == '__main__':
    main()