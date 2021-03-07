import numpy as np
import pandas as pd
import glob
import os

path = './'
documents = ['photos']
datasets = {}

for doc in documents:
    files = glob.glob(path + doc + ".tsv*")
    subsets = []
    for filename in files:
        df = pd.read_csv(filename, sep='\t', header=0)
        subsets.append(df)
        datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)

data = np.array(datasets['photos'])
for idx in range(10000):
    xid = data[idx, 0]
    url = data[idx, 2]
    regular_url = url + "?ixid=" + xid + "\&fm=jpg\&fit=crop\&w=512\&h=512\&q=80\&monochrome=808080"
    cmd = "wget " + regular_url + " -O " + str(idx).zfill(5) + ".jpg"
    print(str(idx).zfill(5), xid)
    os.system(cmd)
