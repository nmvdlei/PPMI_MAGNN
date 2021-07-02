import os
import pickle
from pathlib import Path

dir_data_basename = 'Programming'

def cur_dir():
    return Path().parent.absolute()

def dir_up(path):
    return Path(os.path.dirname(path))

def dir_name(path):
    return os.path.basename(path)

class RetrieveDataException(Exception):
    """Raised when datafolder cannot be found"""
    def __init__(self, dir_data_basename, dir_data_name):
        self.dir_data_basename = dir_data_basename
        self.dir_data_name = dir_data_name
    
    def __str__(self):
        return f'Required "Data" directory cannot be found in your file system. \n'+\
                f'Please manually download "{self.dir_data_name}.7zip" from Dropbox and manually extract in {self.dir_data_basename}'

def find_data_dir(dir_data_basename, dir_data_name='Data'):
    target_dir = cur_dir()
    while dir_name(target_dir) != dir_data_basename:
        if target_dir == dir_up(target_dir):
            raise RetrieveDataException(dir_data_basename, dir_data_name)
        target_dir = dir_up(target_dir)
        
    data_dir = Path(target_dir, dir_data_name)
    if not os.path.exists(data_dir):
        raise RetrieveDataException(dir_data_basename, dir_data_name)
    return data_dir

def dump_in_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def get_file_path(data_dir, data_category, data_format, file_name=None):
    dir_path = Path(data_dir, data_category, data_format)
    
    if file_name:
        return Path(dir_path, file_name) 
    else:
        files = os.listdir(dir_path)
        if len(files)>0:
            return Path(dir_path, files[0])    