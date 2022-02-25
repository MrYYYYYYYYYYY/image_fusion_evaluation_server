import yaml
import os
import numpy as np
lis  = [i[:-8] for i in os.listdir('./dataset/multi-focus/mfif/double')]
lis = np.unique(lis)[1:]
with open('./poses.yaml','w') as dumpfile:
    dumpfile.write(yaml.dump(lis.tolist()))

