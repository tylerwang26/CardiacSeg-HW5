import os
import sys

# Patch os.environ to handle integer values (fix for nnUNetv2 bug)
# The error "TypeError: str expected, not int" occurs because nnUNet tries to set 
# os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1 (int) instead of "1" (str).
def patch_os_environ():
    try:
        # Get the class of os.environ (usually os._Environ)
        environ_cls = os.environ.__class__
        original_setitem = environ_cls.__setitem__

        def safe_setitem(self, key, value):
            if isinstance(value, int):
                value = str(value)
            return original_setitem(self, key, value)

        environ_cls.__setitem__ = safe_setitem
    except Exception as e:
        print(f"Warning: Failed to patch os.environ: {e}")

patch_os_environ()

# Now import nnUNet. The module-level code in run_training.py will execute now.
# Because we patched os.environ, the assignment os.environ['...'] = 1 will succeed (converted to "1").
from nnunetv2.run.run_training import run_training_entry

if __name__ == '__main__':
    run_training_entry()
