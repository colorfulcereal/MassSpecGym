#  python MassSpecGym/scripts/detect_PFAS.py

from massspecgym.models.pfas import HalogenDetectorDreamsTest
ckpt_path = '/teamspace/studios/this_studio/HalogenDetection-FocalLoss-MergedMassSpecNIST20_NISTNew_NormalPFAS/ujmvyfxm/checkpoints/epoch=0-step=9285.ckpt'
model = HalogenDetectorDreamsTest.load_from_checkpoint(ckpt_path)
print(model)

from pathlib import Path
from tqdm import tqdm
from dreams.utils.data import MSData
from dreams.api import dreams_predictions, PreTrainedModel
from dreams.models.heads.heads import BinClassificationHead
from dreams.utils.io import append_to_stem
import pandas as pd
import numpy as np
from pathlib import Path
from dreams.utils.dformats import DataFormatA
from dreams.utils.data import MSData
from dreams.utils.io import append_to_stem
import torch

def find_PFAS(in_pth):
    # in_pth = 'data/teo/<in_file>.mgf'  # or .mzML
    # out_csv_pth = 'data/teo/<in_file>_f_preds.csv'

    # in_pth = Path('/teamspace/studios/this_studio/SLI23_040.mzML')

    n_highest_peaks = 60

    print(f'Processing {in_pth}...')

    # Load data
    try:
        msdata = MSData.from_mzml(in_pth, verbose_parser=True)
    except ValueError as e:
        print(f'Skipping {in_pth} because of {e}.')
        return

    # Get spectra (m/z and inetsnity arrays) and precursor m/z values from the input dataset
    spectra = msdata['spectrum']
    prec_mzs = msdata['precursor_mz']

    # Ref: https://dreams-docs.readthedocs.io/en/latest/tutorials/spectral_quality.html
    # Subject each spectrum to spectral quality checks
    dformat = DataFormatA()
    quality_lvls = [dformat.val_spec(s, p, return_problems=True) for s, p in zip(spectra, prec_mzs)]

    # Check how many spectra passed all filters (`All checks passed`) and how many spectra did not pass some of the filters
    print(pd.Series(quality_lvls).value_counts())

    # Define path for output high-quality file
    hq_pth = append_to_stem(in_pth, 'high_quality').with_suffix('.hdf5')

    # Pick only high-quality spectra and save them to `hq_pth`
    msdata.form_subset(
        idx=np.where(np.array(quality_lvls) == 'All checks passed')[0],
        out_pth=hq_pth
    )

    # Try reading the new file
    msdata_hq = MSData.load(hq_pth)

    # Compute fluorine probabilties
    df = msdata_hq.to_pandas()
    
    f_preds = dreams_predictions(
        spectra=msdata_hq,
        model_ckpt=model,
        n_highest_peaks=n_highest_peaks
    )

    df[f'PFAS_preds'] = torch.sigmoid(torch.from_numpy(f_preds)).cpu().numpy()


    # Store predictions
    # df.to_csv(append_to_stem(in_pth, 'PFAS_preds').with_suffix('.csv'), index=False)
    return df

import os
import pandas as pd

def scan_and_run_pfas(directory, output_csv="pfas_hits.csv", threshold=0.95):
    """
    Scan directory for .mzML files, run find_PFAS() on each, 
    filter predictions, aggregate results, and save to CSV.
    """
    all_hits = []   # list of DataFrames
    num_spectra = 0

    # Loop over all files in directory
    for fname in os.listdir(directory):
        if fname.lower().endswith(".mzml"):
            file_path = os.path.join(directory, fname)
            print(f"Processing: {file_path}")

            try:
                # Call your PFAS detection function
                df = find_PFAS(Path(file_path))   # must return a pandas DataFrame

                # Confirm required column exists
                if "PFAS_preds" not in df.columns:
                    print(f"  ⚠ Warning: no PFAS_preds column in {fname}, skipping.")
                    continue
            
                num_spectra = num_spectra + len(df)

                # Filter based on threshold
                df_hits = df[df["PFAS_preds"] >= threshold].copy()

                # Add file path reference
                df_hits["file_path"] = file_path

                # Only append if non-empty
                if not df_hits.empty:
                    all_hits.append(df_hits)

            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")

    BOLD = '\033[1m'
    END = '\033[0m'
    print(f"\n-----Scanned {BOLD}{num_spectra} entries.{END}------")

    # Combine all records
    if all_hits:
        final_df = pd.concat(all_hits, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"\n✨ Done. Found {len(final_df)} PFAS-like entries.")
        print(f"Output saved to: {output_csv}")
        return final_df
    else:
        print("\n🚫 No PFAS candidates found in any file.")
        return pd.DataFrame()  # empty


# Example usage:
output_csv = '/teamspace/studios/this_studio/mzML_files/pfas_hits.csv'
file_path = '/teamspace/studios/this_studio/mzML_files/Moorea24_MSRun_mzml'
final_results = scan_and_run_pfas(file_path, output_csv=output_csv, threshold=0.9)