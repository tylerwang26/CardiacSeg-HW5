import os
import numpy as np
from pathlib import Path
base = Path(r"c:\CardiacSeg\nnUNet_preprocessed\Dataset001_CardiacSeg")
npz_paths = list(base.rglob('*.npz'))
print(f'Found {len(npz_paths)} .npz files under {base}')
corrupted = []
for p in sorted(npz_paths):
    try:
        with np.load(p) as d:
            # try reading a small array
            _ = d[next(iter(d.files))]
    except Exception as e:
        corrupted.append((str(p), repr(e)))

if corrupted:
    print('\nCorrupted files:')
    for f, e in corrupted:
        print(f'- {f}: {e}')
else:
    print('\nNo corrupted .npz files found')
