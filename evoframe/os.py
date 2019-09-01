import dill as pickle
import os
import shutil

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def pickle_save(obj, filename='obj.pkl'):
    directories = "/".join(filename.split("/")[:-1])
    maybe_make_dir(directories)
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def pickle_load(filename='obj.pkl'):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def maybe_delete_dir(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
