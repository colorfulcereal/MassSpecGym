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
import os
import pandas as pd

def find_PFAS(in_pth):

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

    return df

def scan_and_run_pfas(directory, output_csv="pfas_hits.csv", threshold=0.95):
    """
    Scan directory for .mzML files, run find_PFAS() on each, 
    filter predictions, aggregate results, and save to CSV.
    """
    all_hits = []   # list of DataFrames

    # Loop over all files in directory
    total_len = 0
    for fname in os.listdir(directory):
        if fname.lower().endswith(".mzml"):
            file_path = os.path.join(directory, fname)

            try:
                # Call your PFAS detection function
                df = find_PFAS(Path(file_path))   # must return a pandas DataFrame

                # Confirm required column exists
                if "PFAS_preds" not in df.columns:
                    total_len = total_len + len(df)
                    print(f"  ⚠ Warning: no PFAS_preds column in {fname}, skipping....spectra seen so far..{total_len}")
                    continue

                # Filter based on threshold
                df_hits = df[df["PFAS_preds"] >= threshold].copy()

                # Add file path reference
                df_hits["file_path"] = file_path

                # Only append if non-empty
                if not df_hits.empty:
                    all_hits.append(df_hits)

            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")


    print(f"Total spectra: {total_len}")
    with open("total_spectra_len.txt", 'w') as file_object:
        file_object.write(total_len)

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
final_results = scan_and_run_pfas("/teamspace/studios/this_studio/Moorea24_MSRun_mzml/")