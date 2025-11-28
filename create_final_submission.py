import os
import zipfile
from pathlib import Path
import glob

# Set paths
BASE_DIR = Path(r'C:\CardiacSeg')
INFERENCE_DIR = BASE_DIR / 'inference_3d_lowres_ensemble_test'
SUBMISSION_DIR = BASE_DIR / 'submission_final_ensemble'
ZIP_FILE = BASE_DIR / 'submission_final_ensemble_20251125.zip'

def create_submission():
    print(f"Creating submission from: {INFERENCE_DIR}")
    
    # Create submission directory
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
        
    # Get all prediction files
    pred_files = glob.glob(os.path.join(INFERENCE_DIR, "patient*.nii.gz"))
    print(f"Found {len(pred_files)} prediction files.")
    
    if len(pred_files) == 0:
        print("Error: No prediction files found!")
        return

    # Create ZIP file
    print(f"Zipping files to: {ZIP_FILE}")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in pred_files:
            file_name = os.path.basename(file_path)
            # Add file to zip
            zipf.write(file_path, file_name)
            print(f"Added: {file_name}")
            
    print("\nSubmission ZIP created successfully!")
    print(f"Location: {ZIP_FILE}")
    print("You can now upload this file to the competition platform.")

if __name__ == "__main__":
    create_submission()
