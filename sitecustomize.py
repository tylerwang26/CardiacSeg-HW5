# Make Python 3.12+/3.13+ compatible with packages importing 'distutils'
# by redirecting to setuptools' vendored implementation when available.
# This file is auto-imported by Python's site module if present on sys.path.
try:
    import importlib
    import sys
    # Try to import setuptools vendored distutils
    _vendored = importlib.import_module('setuptools._distutils')
    sys.modules.setdefault('distutils', _vendored)
    # Also provide submodules commonly imported
    for submod in ('file_util', 'dir_util', 'dist', 'spawn', 'sysconfig'):
        try:
            sys.modules.setdefault(f'distutils.{submod}', importlib.import_module(f'setuptools._distutils.{submod}'))
        except Exception:
            pass
except Exception:
    # If anything fails, leave environment unchanged
    pass
