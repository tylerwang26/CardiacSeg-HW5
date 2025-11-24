# Compatibility shim for Python 3.12+ where stdlib 'distutils' is removed.
# Redirect imports to setuptools' vendored implementation if available.
import importlib, sys

try:
    _vendored = importlib.import_module('setuptools._distutils')
except Exception as e:  # Fallback: raise informative error
    raise ImportError("distutils is not available; please install setuptools or run within the project root so this shim can work.") from e

# Populate expected attributes from vendored module
globals().update({k: getattr(_vendored, k) for k in dir(_vendored) if not k.startswith('_')})

# Ensure submodules resolve, e.g., 'from distutils.file_util import copy_file'
for _sub in ('file_util', 'dir_util', 'dist', 'spawn', 'sysconfig', 'log', 'util'):
    try:
        _m = importlib.import_module(f'setuptools._distutils.{_sub}')
        sys.modules.setdefault(f'distutils.{_sub}', _m)
    except Exception:
        pass
