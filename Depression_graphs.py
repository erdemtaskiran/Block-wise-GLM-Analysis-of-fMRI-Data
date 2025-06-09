import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting

def main():
    # Set up directories
    root = "/Users/erdemtaskiran/Desktop/Depression/Depression"
    betas_dir = os.path.join(root, "Betas_all_blocks", "last_beta_maps_true")
    plots_dir = os.path.join(root, "Design_Matrix_Plots")
    
    # Create output directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get all subject directories
    sub_dirs = [d for d in os.listdir(betas_dir) if d.startswith('sub-')]
    print(f"Found {len(sub_dirs)} subject directories")
    
    for sub in sub_dirs:
        print(f"\nProcessing subject: {sub}")
        sub_dir = os.path.join(betas_dir, sub)
        print(f"Looking in directory: {sub_dir}")
        
        # List directory contents
        print("Directory contents:")
        print(os.listdir(sub_dir))
        
        # Find all design matrix files
        design_files = glob.glob(os.path.join(sub_dir, "*_design.tsv"))
        print(f"Found {len(design_files)} design matrix files\n")
        
        for design_file in design_files:
            try:
                # Read design matrix
                design = pd.read_csv(design_file, sep='\t')
                print(f"Processing design file: {design_file}")
                print(f"Design matrix shape: {design.shape}")
                print(f"Design matrix columns: {list(design.columns)}")
                
                # Extract task type and run number from filename
                filename = os.path.basename(design_file)
                task_type = filename.split('_')[1].split('-')[1]
                run = filename.split('_')[2].split('-')[1]
                print(f"  Plotting design matrix for {task_type} run {run}")
                
                # Create figure with title
                plt.figure(figsize=(15, 10))
                plotting.plot_design_matrix(design)
                plt.title(f'{sub} {task_type} run-{run} design')
                
                # Save plot
                output_file = os.path.join(plots_dir, f"{sub}_task-{task_type}_run-{run}_design.png")
                plt.savefig(output_file)
                plt.close()
                
            except Exception as e:
                print(f"Error processing {design_file}: {str(e)}")
                import traceback
                print(traceback.format_exc())
    
    print(f"\nFinished – Design matrix plots written to {plots_dir}")

if __name__ == "__main__":
    main() 