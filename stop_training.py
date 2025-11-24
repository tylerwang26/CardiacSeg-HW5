import psutil
import sys
import os
import signal

def kill_training_processes():
    killed = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in proc.info['name']:
                # Check for the monitoring script or the actual training
                if any('train_3d_fullres_monitored.py' in arg for arg in cmdline) or \
                   any('nnunetv2.run.run_training' in arg for arg in cmdline):
                    print(f"Found process: {proc.info['pid']} {' '.join(cmdline)}")
                    print(f"Killing process {proc.info['pid']}...")
                    proc.kill()
                    killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed:
        print("Training processes terminated.")
    else:
        print("No training processes found.")

if __name__ == "__main__":
    # Install psutil if not present (it should be in the venv usually, but let's check)
    try:
        import psutil
        kill_training_processes()
    except ImportError:
        print("psutil not found. Please install it or kill processes manually.")
