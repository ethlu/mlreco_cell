import os, sys
import shutil
import numpy as np

def move_processed(key, in_dir, processed_dir, move_dir):
    processed = set([key(f) for f in os.listdir(processed_dir)])  
    print(processed)
    print("len processed ", len(processed))
    target = [f for f in os.listdir(in_dir) if key(f) in processed]
    print(target)
    print("len target ", len(target))
    for t in target:
        shutil.move(in_dir+'/'+t, move_dir+'/'+t)

def partition(key, n, in_dir, out_dir):
    n = int(n)
    f_dict = {}
    for f in os.listdir(in_dir):
        try:
            f_dict[key(f)].append(f)
        except:
            f_dict[key(f)] = [f]
    print(f_dict)
    len_sample = len(f_dict.keys())
    print("len samples ", len_sample)
    if len_sample % n == 0:
        per_part = len_sample//n
        i, j = 0, 0
        os.mkdir(out_dir+'/%d'%j)
        for v in f_dict.values():
            if i==per_part:
                i = 0
                j += 1
                os.mkdir(out_dir+'/%d'%j)
            for f in v:
                shutil.copy(in_dir+'/'+f, out_dir+'/%d/'%j+f)
            i += 1
    else:
        per_part = len_sample//n
        start_i = n - (len_sample % n)
        i, j = 0, 0
        os.mkdir(out_dir+'/%d'%j)
        for v in f_dict.values():
            if (i < start_i and i==per_part) or (i >= start_i and i==per_part+1):
                i = 0
                j += 1
                os.mkdir(out_dir+'/%d'%j)
            for f in v:
                shutil.copy(in_dir+'/'+f, out_dir+'/%d/'%j+f)
            i += 1
        
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

KEYS = {
        "MU" : lambda f: int(f[f.find("Mu")+3: f.find("Mu")+6])
        }

def sort_data_dir(data_dir, key_name = None):
    if key_name is None:
        return sorted(os.listdir(data_dir))
    return sorted(os.listdir(data_dir), key = KEYS[key_name])

if __name__=="__main__":
    #print(sort_data_dir(sys.argv[1], "MU"))
    pass
    #move_processed(lambda f: f[f.find("Mu"): f.find("Mu")+6])
    #partition(lambda f: f[f.find("Mu"): f.find("Mu")+6])
    #partition(lambda f: f[0]) 
    #print(n_splits(8, 4))
