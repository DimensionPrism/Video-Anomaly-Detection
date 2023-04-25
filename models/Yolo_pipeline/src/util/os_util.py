import os
import shutil

from os import listdir, unlink
from os.path import join, exists, isfile, islink, isdir
from shutil import copy, move, rmtree

def remove_tree(dir, keep_dir=True):
    print(f"Deleting directory: '{dir}'. Keep directory: {keep_dir}")
    if not exists(dir):
        raise Exception(f"Directory '{dir}' not exists!")
    if keep_dir:
        for filename in listdir(dir):
            file_path = join(dir, filename)
            try:
                if isfile(file_path) or islink(file_path):
                    unlink(file_path)
                elif isdir(file_path):
                    rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        rmtree(dir)