import os
import multiprocessing

def run_exe(arguments):
    cmd = 'python ../main.py ' + arguments
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    processes = []

    instances = []
    instances = [
                '../benchmarks/gdb/gdb2.dat',
                '../benchmarks/gdb/gdb1.dat'
                ]
    instance_path = "../bridge-graph-instances/standard/howe/"
    # loop through all the files in the directory and print out the file names
    # for file_name in os.listdir(instance_path):
    #     instances.append(instance_path+file_name)

    # seeds
    seeds = "8115,3520,8647,9420,3116,6377,6207,4187,3641,8591,3580,8524,2650,2811,9963,7537,3472,3714,8158,7284,6948,6119,5253,5134,7350,2652,9968,3914,6899,4715"

    # depot configurations / k-values
    k_depots = [[0,0]] # [[0,0],[0,0,0,0],[0,0,0,0,0,0,0,0]]

    # heuristics
    heuristics = ['MMMR', 'RR']

    print("Running " + str(len(instances) * len(k_depots) * len(heuristics)) + " different configurations.")

    # create a Pool object with the number of available processors
    pool = multiprocessing.Pool(processes=num_cores)

    for k in k_depots:
        depots = ""
        for d in k:
            depots += str(d) + ","
        depots = depots[:-1]
        for instance in instances:
            if '.obj' in instance:
                continue
            for heuristic in heuristics:
                arguments = "-i " + instance + " -k " + depots + " -s " + seeds + " -j " + heuristic
                print(arguments)
                pool.apply_async(run_exe, (arguments,))

                p = multiprocessing.Process(target=run_exe, args=(arguments,))
                processes.append(p)
                p.start()

    # close the pool to prevent any more tasks from being added
    pool.close()

    # wait for all the processes to finish
    pool.join()