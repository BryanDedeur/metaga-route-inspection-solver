import os
import multiprocessing

def run_exe(arguments):
    cmd = 'python ../main.py ' + arguments
    print(cmd)
    os.system(cmd)

def find_files_with_extension(dir, extensions):
    """
    Recursively find all files in root_dir and its subdirectories
    that have one of the extensions in the extensions list.
    
    Args:
    - root_dir: string, the root directory to start searching in.
    - extensions: list of strings, the extensions to filter files by.
    
    Returns:
    - list of strings, the absolute paths of all matching files.
    """
    # Initialize an empty list to hold the file paths.
    file_paths = []
    
    # Loop over all files and directories in root_dir.
    for entry in os.scandir(dir):
        # If the entry is a directory, recursively call this function
        # with the directory as the new root_dir.
        if entry.is_dir():
            file_paths.extend(find_files_with_extension(entry.path, extensions))
        # If the entry is a file and its extension matches one of the
        # extensions in the extensions list, add its path to the file_paths list.
        elif entry.is_file() and os.path.splitext(entry.name)[1] in extensions:
            file_paths.append(entry.path)
    
    # Return the list of matching file paths.
    return file_paths

if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count() - 1
    processes = []

    extensions = ('.dat', '.csv')
    instances = find_files_with_extension("../benchmark_subset", extensions)

    # seeds
    seeds = "0"

    # depot configurations / k-values
    k_depots = [[0,0],[0,0,0,0],[0,0,0,0,0,0,0,0]]

    # heuristics
    heuristics = ['MIN-MEDIAN', 'MIN-MAX']

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
                pool.apply_async(run_exe, (arguments,))

    # close the pool to prevent any more tasks from being added
    pool.close()

    # wait for all the processes to finish
    pool.join()