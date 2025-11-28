
import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    result = subprocess.run([sys.executable, script_name], check=True)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        sys.exit(1)

def main():
    # 1. Generate 3D Lowres 5-Fold Probabilities
    if not os.path.exists('inference_3d_lowres_5fold_prob'):
        run_script('run_inference_3d_lowres_5fold.py')
    else:
        print("inference_3d_lowres_5fold_prob already exists. Skipping inference.")

    # 2. Generate V15 Submission
    run_script('run_v15_submission.py')
    
    # 3. Generate V16 Submission
    # Ensure Fullres Fold 1 is ready (it should be, based on file check)
    run_script('run_v16_submission.py')
    
    print("\n" + "="*50)
    print("V15 and V16 Submissions Generated Successfully!")
    print("="*50)
    print("V15: submission_final_v15_20251126.zip")
    print("V16: submission_final_v16_20251126.zip")
    print("\nNote regarding V17:")
    print("V17 requires 5-Fold ensembles for ALL models (2D, 3D Lowres, 3D Fullres).")
    print("Currently, we are missing:")
    print("- 2D Folds 1-4")
    print("- 3D Fullres Folds 2-4")
    print("Training these models will take significant time (hours).")
    print("Please upload V15 and V16 first.")

if __name__ == '__main__':
    main()
