import os, glob, json, numpy as np, pandas as pd, nibabel as nib
from nilearn.glm.first_level import FirstLevelModel

# ---- CONFIG -----------------------------------------------------------------
root   = "/Users/erdemtaskiran/Desktop/Depression/Depression"
# Create main output directory for all subjects with block-wise analysis
betas_all_dir = os.path.join(root, "Betas_all_blocks", "last_beta_maps_true")  # Updated directory path
os.makedirs(betas_all_dir, exist_ok=True)

# Get all subject directories
sub_dirs = sorted([d for d in os.listdir(root) if d.startswith('sub-')])
print(f"Found {len(sub_dirs)} subject directories: {sub_dirs}")

# Define task types and their corresponding trial types
task_types = {
    'music': ['positive_music', 'negative_music'],
    'nonmusic': ['positive_nonmusic', 'negative_nonmusic']
}

# -----------------------------------------------------------------------------

# Process each subject
for sub in sub_dirs:
    print(f"\nProcessing subject: {sub}")
    func = os.path.join(root, sub, "func")     # BIDS-style location
    outdir = os.path.join(betas_all_dir, sub)  # where β-maps will be saved
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Looking for BOLD files in: {func}")
    if not os.path.exists(func):
        print(f"Error: func directory does not exist: {func}")
        continue

    # Process each task type (music and non-music)
    for task_type in task_types.keys():
        print(f"\nProcessing {task_type} task")
        
        # Find all BOLD files for this task type - ONLY using wasub images
        bold_patterns = [
            f"wasub-*_task-{task_type}_run-*_bold.nii",
            f"wasub-*_task-{task_type}_run-*_bold.nii.gz"
        ]
        
        bold_files = []
        for pattern in bold_patterns:
            bold_files.extend(glob.glob(os.path.join(func, pattern)))
        
        bold_files = sorted(bold_files)
        print(f"Found {len(bold_files)} wasub BOLD files for {task_type} task")
        
        if not bold_files:
            print(f"Warning: No wasub BOLD files found for {sub} {task_type} task")
            continue

        for run_idx, bold in enumerate(bold_files, start=1):
            print(f"  Processing run {run_idx}/{len(bold_files)}")
            print(f"  BOLD file: {bold}")
            
            # Get the run number from the BOLD filename
            bold_basename = os.path.basename(bold)
            run = bold_basename.split('_run-')[1].split('_')[0]
            
            # Find corresponding events file
            events_pattern = f"sub-{sub.split('-')[1]}_task-{task_type}_run-{run}_events.tsv"
            events = os.path.join(func, events_pattern)
            
            # Find corresponding motion parameter file using glob
            conf_pattern = os.path.join(func, f"rp_asub-*_task-{task_type}_run-{run}_bold.txt")
            candidates = glob.glob(conf_pattern)
            if not candidates:
                print(f"  Error: No motion parameter file found matching pattern: {conf_pattern}")
                continue
            confound = candidates[0]
            
            print(f"  Events file: {events}")
            print(f"  Confound file: {confound}")
            
            if not os.path.exists(events):
                print(f"  Error: Events file not found: {events}")
                continue

            # read TR from the task-specific JSON file
            json_file = os.path.join(func, f"task-{task_type}_bold.json")
            if os.path.exists(json_file):
                TR = json.load(open(json_file))["RepetitionTime"]
                print(f"  TR from JSON: {TR}")
            else:
                TR = 3.0   # fallback
                print(f"  Using default TR: {TR}")

            # ---------------------------------------------------------------- build GLM
            fmri_glm = FirstLevelModel(
                t_r=TR,
                hrf_model="spm",
                high_pass=1/128,       # 128-s cut-off like SPM
                smoothing_fwhm=None,   # we WANT unsmoothed data
                mask_img=None,         # Nilearn will compute an epi mask
                minimize_memory=True,
            )

            try:
                # read events.tsv
                ev = pd.read_csv(events, sep="\t")
                print(f"  Events file columns: {ev.columns.tolist()}")
                print(f"  Unique trial types: {ev['trial_type'].unique()}")
                
                # Filter for the relevant trial types for this task
                ev = ev[ev.trial_type.isin(task_types[task_type])].reset_index(drop=True)
                print(f"  Number of trials after filtering: {len(ev)}")

                # Create separate blocks for each condition - using groupby to ensure proper block numbering
                ev_list = []
                for cond, grp in ev.groupby('trial_type'):
                    for k, row in grp.reset_index(drop=True).iterrows():
                        block_name = f"{cond}_block{k+1}"
                        ev_list.append(
                            dict(onset=row.onset,
                                 duration=row.duration,
                                 trial_type=block_name)
                        )
                ev = pd.DataFrame(ev_list)
                print(f"  Created {len(ev)} separate blocks")
                print(f"  Block names: {ev['trial_type'].unique()}")

                # confounds: the 6-column realignment file from SPM
                confounds = pd.read_csv(confound, sep="\s+", header=None,
                                    names=[f"motion_{i}" for i in range(6)])
                print(f"  Confounds shape: {confounds.shape}")

                print("  Fitting GLM...")
                fmri_glm.fit(bold, events=ev, confounds=confounds)
                print("  GLM fit completed")

                design = fmri_glm.design_matrices_[0]
                print(f"  Design matrix shape: {design.shape}")
                print(f"  Design columns: {design.columns.tolist()}")  # Sanity check for block names
                
                # Save design matrix
                design_path = os.path.join(outdir, f"{sub}_task-{task_type}_run-{run}_design.tsv")
                design.to_csv(design_path, sep='\t')
                print(f"  Saved design matrix to: {design_path}")

                # ---------------------------------------------------------------- extract β
                for cond in ev.trial_type.unique():
                    print(f"    Computing contrast for: {cond}")
                    # one-hot contrast for *this* column
                    con = np.zeros(len(design.columns))
                    con[design.columns.get_loc(cond)] = 1.0

                    beta_map = fmri_glm.compute_contrast(con,
                                                     output_type="effect_size")
                    print(f"    Beta map shape: {beta_map.shape}")

                    # file name encodes everything CosmoMVPA needs
                    beta_fname = f"{sub}_task-{task_type}_run-{run}_{cond}_beta.nii.gz"
                    output_path = os.path.join(outdir, beta_fname)
                    nib.save(beta_map, output_path)
                    print(f"    Saved beta map to: {output_path}")
                    
            except Exception as e:
                print(f"Error processing {sub} {task_type} run {run}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue

print(f"\nFinished – β-maps written to {betas_all_dir}")
