import os
import urllib
import tarfile
from datetime import datetime 

import numpy as np

def fetch_data(url, dirpath, filename=None, force=False):
    if not os.listdir(dirpath) or force is True:
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        if not filename:
            filename = f'data_{_get_timestamp()}'
        filepath = os.path.join(dirpath, filename)
        urllib.request.urlretrieve(url, filepath)
        return filename

def extract_tarfile(path, filename):
    archive = tarfile.open(os.path.join(path, filename))
    archive.extractall(path=path)
    archive.close()

def _get_timestamp():
    now = datetime.now(tz=None)
    return f"{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}"
