import numpy as np
import sys
p=r'C:\CardiacSeg\nnUNet_preprocessed\Dataset001_CardiacSeg\nnUNetPlans_2d\patient0001.npz'
print('loading',p)
try:
    a=np.load(p)
    print('keys',list(a.keys()))
except Exception as e:
    print('ERROR',repr(e))
    sys.exit(1)
print('loaded OK')
