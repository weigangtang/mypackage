import numpy as np
import pandas as pd

import os
import glob

def sort_by_list(A, B):
    # elements in A were sorted according to the order of B
    # elements in A but not in B will be ignored
    sorted_A = [item for item in B if item in A]
    if len(sorted_A) < len(A): 
        print('{:d} items were ignored.'.format(len(A)-len(sorted_A))) 
        print(list(set(A).difference(sorted_A)))
    return sorted_A

def get_prev_dir(path, back=1):
    return '/'.join(path.split('/')[:-back])

def check_if_file_exist(path):
    if not os.path.exists(path): 
        print('Can not found ' + path)

def extract_keywords(fpath_frame, ref_list=None):
    files = glob.glob(fpath_frame.format('*'))
    p1, p2 = fpath_frame.split('{}')
    keyword_list = [f.replace(p1, '').replace(p2, '') for f in files]
    if ref_list is None:
        keyword_list = np.sort(keyword_list).tolist()
    else:
        keyword_list = sort_by_list(keyword_list, ref_list)
    return keyword_list

def search_scripts_with_keyword(keyword, folder, file_type='py'):
    item_list = glob.glob(folder + '/*')
    for item in item_list:
        if os.path.isdir(item):
            search_scripts_with_keyword(keyword, item, file_type) # recursive
        else:
            if item.split('.')[-1] == file_type:
                with open(item) as f: 
                    if keyword in f.read():
                        print(item)

def search_scripts_with_keywords(keywords, folder, file_type='py'):
    item_list = glob.glob(folder + '/*')
    for item in item_list:
        if os.path.isdir(item):
            search_scripts_with_keywords(keywords, item, file_type) # recursive
        else:
            if item.split('.')[-1] == file_type:
                with open(item) as f: 
                    content = f.read()
                    is_all_in = all([kw in content for kw in keywords])
                    if is_all_in: print(item)
