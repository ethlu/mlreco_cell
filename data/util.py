import os, sys
import shutil
import numpy as np

EVENT_TYPES = ["Mu", "Electron", "Pion"]
PRODUCT_TYPES = ["root", "energy", "wire", "hit", "xy", "pixel"]
FILE_REGEX = "(%s).(\d+).(%s)" %("|".join(EVENT_TYPES), "|".join(PRODUCT_TYPES))

def file_info(filename):
    import re
    x = re.findall(FILE_REGEX, filename)
    if len(x) != 1:
        print("bad file name ", filename)
        return None
    evt, index, prod = x[0]
    return evt, int(index), prod

def inf_file_info(filename):
    import re
    x = re.findall("epoch(\d+).(.+).np.", filename)
    if len(x) != 1:
        print("bad file name ", filename)
        return None
    epoch, f = x[0]
    return int(epoch), f

def time_info(filename):
    return os.stat(filename)[-2],

def files_info(files, groupby=[1], file_info=file_info):
    files_dict = {}
    for f in files:
        info = file_info(f)
        if info is None: continue
        key = tuple([info[i] for i in groupby])
        try:
            files_dict[key].append(f)
        except KeyError:
            files_dict[key] = [f]
    return files_dict

def filter_fd(files_dict, func=lambda k, v: True):
    filtered_dict = {}
    for k, v in files_dict.items():
        if func(k, v):
            filtered_dict[k] = v
    return filtered_dict

def diff_fd(dictA, dictB):
    return filter_fd(dictA, lambda k, v: k not in dictB)

def flatten_fd(files_dict):
    sorted_indices = sorted(files_dict.keys()) 
    samples = []
    for i in sorted_indices:
        samples.extend(files_dict[i])
    return samples
        
def n_splits(len_sample, n):
    per_part = len_sample//n
    if len_sample % n == 0:
        return np.arange(n+1)*per_part
    else:
        start_i = n - (len_sample % n)
        splits = [0]
        for i in range(n):
            if i>=start_i:
                splits.append(splits[-1]+per_part+1)
            else:
                splits.append(splits[-1]+per_part)
        return splits

def slurm_split(samples):
    n_tasks = os.getenv('SLURM_NTASKS')
    if n_tasks is not None:
        n_tasks = int(n_tasks)
        len_sample = len(samples)
        splits = n_splits(len_sample, n_tasks)
        rank = int(os.getenv('SLURM_PROCID'))
        samples = samples[splits[rank]:splits[rank+1]]
    return samples

def remove_latest(dir_name):
    WITHIN = 10
    files = [dir_name+'/'+f for f in os.listdir(dir_name)]
    time_dict = files_info(files, [0], time_info)
    latest_time = max(time_dict.keys())[0]
    remove = flatten_fd(filter_fd(time_dict, lambda k, v: k[0] > latest_time - WITHIN))
    print("Files removed: ", remove)
    print("Number: ",len(remove))
    for f in remove:
        os.remove(f)

if __name__=="__main__":
    #file_info(sys.argv[1])
    #remove_latest(sys.argv[1])
    #inf_file_info(sys.argv[1])
    pass
    #move_processed(lambda f: f[f.find("Mu"): f.find("Mu")+6])
    #partition(lambda f: f[f.find("Mu"): f.find("Mu")+6])
    #partition(lambda f: f[0]) 
    #print(n_splits(8, 4))
