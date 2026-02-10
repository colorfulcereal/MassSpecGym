#!/usr/bin/env python3
"""
PFAS Identification CLI Tool

Implements the MassQL workflow for PFAS detection:
- CF2 losses detection (perfluoroalkyl chains)
- Diagnostic fragment matching
- Kendrick Mass Defect analysis (CF2-based)
- Mass Defect filtering (fluorine abundance pattern)

Usage:
    python scripts/pfas_identification.py \\
        --input ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \\
        --fold val \\
        --output results/pfas_predictions.tsv \\
        --report results/pfas_report.txt
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data(filepath, fold):
    """Load TSV and filter to specified fold."""
    print(f"Loading {fold} fold from {filepath}...")
    df = pd.read_csv(filepath, sep='\t')
    df_fold = df[df['fold'] == fold].copy()
    return df_fold


# =============================================================================
# MassQL Detection Functions
# =============================================================================

def find_peak_within_tolerance(target_mz, mz_list, ppm_tol):
    """Check if target m/z exists in spectrum within tolerance."""
    for mz in mz_list:
        ppm_error = abs((mz - target_mz) / target_mz) * 1e6
        if ppm_error <= ppm_tol:
            return True
    return False


def detect_cf2_losses(mzs, intensities, precursor_mz, ppm_tol=10):
    """
    Step 2a: Detect perfluoroalkyl chains via CF2 losses.

    Look for repeated CF2 spacing (49.9968 Da) in fragment pattern.
    Returns score based on number of CF2 units detected.
    """
    CF2_MASS = 49.9968
    cf2_count = 0

    # Check for CF2 ladder in fragments
    for i, mz in enumerate(mzs):
        # Look for mz + n*CF2 patterns
        for n in range(1, 10):
            expected_mz = mz + n * CF2_MASS
            if find_peak_within_tolerance(expected_mz, mzs, ppm_tol):
                cf2_count += 1
                break  # Count each base peak once

    return cf2_count


def detect_diagnostic_fragments(mzs, intensities, ppm_tol=10, min_rel_int=0.01):
    """
    Step 2b: Match diagnostic PFAS fragments.

    Diagnostic fragments from MassQL approach:
    - CF3: 68.9952
    - C2F5: 118.9916
    - C3F5: 131.00
    - C3F7: 168.987
    - SO3: 79.95736 (sulfur-containing PFAS)
    - HSO4: 96.96010
    - FSO3: 98.95576

    Returns dict with matched fragments and score.
    """
    DIAGNOSTIC_FRAGS = {
        'CF3': 68.9952,
        'C2F5': 118.9916,
        'C3F5': 131.00,
        'C3F7': 168.987,
        'SO3': 79.95736,
        'HSO4': 96.96010,
        'FSO3': 98.95576
    }

    if len(intensities) == 0:
        return {'matched_fragments': [], 'fragment_score': 0}

    max_intensity = max(intensities)
    matches = []

    for name, ref_mz in DIAGNOSTIC_FRAGS.items():
        for mz, intensity in zip(mzs, intensities):
            rel_intensity = intensity / max_intensity
            ppm_error = abs((mz - ref_mz) / ref_mz) * 1e6

            if ppm_error <= ppm_tol and rel_intensity >= min_rel_int:
                matches.append(name)
                break

    return {
        'matched_fragments': matches,
        'fragment_score': len(matches)
    }


def calculate_mass_defect(precursor_mz):
    """
    Calculate regular mass defect for PFAS detection.

    Mass defect is the fractional part of the exact mass.
    PFAS compounds typically have mass defects in the range 0.9-0.99
    due to the high fluorine content (F = 18.998 Da).

    Example: 450.9234 Da → mass defect = 0.9234

    This is complementary to Kendrick Mass Defect and is used in
    ion mobility-based PFAS screening (MassQL approach).

    Args:
        precursor_mz: Precursor m/z value

    Returns:
        float: Mass defect (fractional part of mass)
    """
    return precursor_mz - int(precursor_mz)


def calculate_kendrick_mass_defect(precursor_mz, base='CF2'):
    """
    Calculate Kendrick Mass Defect (KMD) for PFAS detection.

    KMD analysis rescales the mass scale to make homologous series with
    repeating units (CF2 for PFAS) align at the same nominal Kendrick mass.

    For CF2-based KMD:
    - Kendrick Mass = observed_mass × (14.00000 / 49.9968)
    - Kendrick Mass Defect = nominal_KM - KM

    PFAS compounds typically have KMD values close to 0 or in specific ranges
    when using CF2 as the base unit.

    Args:
        precursor_mz: Precursor m/z value
        base: Base unit for rescaling (default: 'CF2')

    Returns:
        dict with 'kendrick_mass', 'kmd', 'is_pfas_kmd' (bool)
    """
    BASE_MASSES = {
        'CF2': 49.9968,  # CF2 exact mass
        'CH2': 14.01565  # Alternative base for comparison
    }

    if base not in BASE_MASSES:
        raise ValueError(f"Unknown base: {base}. Use 'CF2' or 'CH2'")

    base_mass = BASE_MASSES[base]

    # Calculate Kendrick mass
    kendrick_mass = precursor_mz * (14.00000 / base_mass)

    # Calculate Kendrick mass defect
    nominal_km = round(kendrick_mass)
    kmd = nominal_km - kendrick_mass

    # PFAS typically have KMD in range [-0.1, 0.1] for CF2 base
    # This range can be tuned based on empirical data
    is_pfas_kmd = abs(kmd) <= 0.15

    return {
        'kendrick_mass': kendrick_mass,
        'kmd': kmd,
        'is_pfas_kmd': is_pfas_kmd
    }


def compute_pfas_score(row, ppm_tol=10, min_intensity=0.01, use_kmd=True, kmd_threshold=0.15,
                       use_mass_defect=True, mass_defect_min=0.9, mass_defect_max=0.99):
    """
    Combine detection methods into overall PFAS score.

    Scoring:
    - CF2 losses: 2 points per detected CF2 unit
    - Diagnostic fragments: 3 points per matched fragment
    - Kendrick Mass Defect: 4 points if |KMD| <= threshold (default 0.15)
    - Mass Defect: 3 points if mass defect in PFAS range (default 0.9-0.99)
    - Threshold: score >= 5 indicates PFAS
    """
    mzs = row['mzs_parsed']
    intensities = row['intensities_parsed']
    precursor = row['precursor_mz']

    # CF2 chain detection
    cf2_score = detect_cf2_losses(mzs, intensities, precursor, ppm_tol) * 2

    # Diagnostic fragment detection
    frag_result = detect_diagnostic_fragments(mzs, intensities, ppm_tol, min_intensity)
    frag_score = frag_result['fragment_score'] * 3

    # Kendrick Mass Defect analysis (if enabled)
    kmd_score = 0
    kmd_value = 0.0
    kendrick_mass = 0.0
    if use_kmd:
        kmd_result = calculate_kendrick_mass_defect(precursor, base='CF2')
        kmd_value = kmd_result['kmd']
        kendrick_mass = kmd_result['kendrick_mass']
        # Award 4 points if KMD indicates PFAS-like pattern
        if abs(kmd_value) <= kmd_threshold:
            kmd_score = 4
    else:
        kmd_value = None
        kendrick_mass = None

    # Mass Defect filtering (complementary to KMD)
    mass_defect_score = 0
    mass_defect_value = calculate_mass_defect(precursor)
    if use_mass_defect:
        # PFAS typically have mass defects between 0.9-0.99 due to fluorine content
        if mass_defect_min <= mass_defect_value <= mass_defect_max:
            mass_defect_score = 3

    total_score = cf2_score + frag_score + kmd_score + mass_defect_score

    return {
        'cf2_score': cf2_score,
        'fragment_score': frag_score,
        'kmd_score': kmd_score,
        'kmd': kmd_value,
        'kendrick_mass': kendrick_mass,
        'mass_defect_score': mass_defect_score,
        'mass_defect': mass_defect_value,
        'matched_fragments': str(frag_result['matched_fragments']),
        'total_score': total_score,
        'predicted_pfas': total_score >= 5
    }


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }


def generate_report(df_results, metrics, args):
    """Generate formatted text report."""
    with open(args.report, 'w') as f:
        f.write("PFAS Identification Results\n")
        f.write("============================\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Fold: {args.fold}\n")
        f.write(f"Method: MassQL\n")
        f.write(f"Samples: {len(df_results)}\n\n")

        f.write("Workflow:\n")
        f.write("MassQL Detection Methods:\n")
        f.write("  - CF2 loss detection (perfluoroalkyl chains)\n")
        f.write("  - Diagnostic fragment matching\n")
        if args.use_kmd:
            f.write(f"  - Kendrick Mass Defect analysis (CF2 base, threshold={args.kmd_threshold})\n")
        else:
            f.write("  - Kendrick Mass Defect: Disabled\n")
        if args.use_mass_defect:
            f.write(f"  - Mass Defect filtering (range: {args.mass_defect_min}-{args.mass_defect_max})\n")
        else:
            f.write("  - Mass Defect filtering: Disabled\n")
        f.write("\n")

        f.write("Parameters:\n")
        f.write(f"- PPM tolerance: {args.ppm_tol}\n")
        f.write(f"- Minimum relative intensity: {args.min_intensity}\n")
        f.write("- CF2 score weight: 2\n")
        f.write("- Fragment score weight: 3\n")
        if args.use_kmd:
            f.write(f"- KMD score weight: 4 (if |KMD| <= {args.kmd_threshold})\n")
        if args.use_mass_defect:
            f.write(f"- Mass Defect score weight: 3 (if {args.mass_defect_min} <= MD <= {args.mass_defect_max})\n")
        f.write("- Score threshold: 5\n\n")

        f.write("Diagnostic Fragments:\n")
        f.write("- CF3: 68.9952\n")
        f.write("- C2F5: 118.9916\n")
        f.write("- C3F5: 131.00\n")
        f.write("- C3F7: 168.987\n")
        f.write("- SO3: 79.95736 (sulfur-containing PFAS)\n")
        f.write("- HSO4: 96.96010\n")
        f.write("- FSO3: 98.95576\n\n")

        f.write("Classification Metrics:\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n\n")

        # Confusion matrix
        y_true = df_results['is_PFAS'].astype(bool)
        y_pred = df_results['predicted_pfas'].astype(bool)
        cm = confusion_matrix(y_true, y_pred)
        f.write("Confusion Matrix:\n")
        f.write("                Predicted PFAS    Predicted Non-PFAS\n")
        f.write(f"Actual PFAS          {cm[1,1]:>4}               {cm[1,0]:>4}\n")
        f.write(f"Actual Non-PFAS       {cm[0,1]:>4}               {cm[0,0]:>4}\n\n")

        # Kendrick Mass Defect statistics
        if args.use_kmd:
            kmd_evidence = (df_results['kmd_score'] > 0).sum()
            kmd_only = ((df_results['cf2_score'] == 0) &
                       (df_results['fragment_score'] == 0) &
                       (df_results['kmd_score'] > 0) &
                       (df_results['predicted_pfas'] == True)).sum()
            avg_kmd_pfas = df_results[df_results['predicted_pfas'] == True]['kmd'].mean()
            avg_kmd_non_pfas = df_results[df_results['predicted_pfas'] == False]['kmd'].mean()
            f.write("Kendrick Mass Defect Analysis:\n")
            f.write(f"- Samples with KMD evidence (|KMD| <= {args.kmd_threshold}): {kmd_evidence}\n")
            f.write(f"- KMD-only predictions (no CF2/fragments): {kmd_only}\n")
            f.write(f"- Average |KMD| for predicted PFAS: {abs(avg_kmd_pfas):.4f}\n")
            f.write(f"- Average |KMD| for predicted non-PFAS: {abs(avg_kmd_non_pfas):.4f}\n\n")

        # Mass Defect statistics
        if args.use_mass_defect:
            md_evidence = (df_results['mass_defect_score'] > 0).sum()
            avg_md_pfas = df_results[df_results['predicted_pfas'] == True]['mass_defect'].mean()
            avg_md_non_pfas = df_results[df_results['predicted_pfas'] == False]['mass_defect'].mean()
            f.write("Mass Defect Analysis:\n")
            f.write(f"- Samples with mass defect in PFAS range ({args.mass_defect_min}-{args.mass_defect_max}): {md_evidence}\n")
            f.write(f"- Average mass defect for predicted PFAS: {avg_md_pfas:.4f}\n")
            f.write(f"- Average mass defect for predicted non-PFAS: {avg_md_non_pfas:.4f}\n\n")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PFAS Identification using MassQL workflow'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input TSV file')
    parser.add_argument('--fold', type=str, default='val',
                       help='Which fold to predict on (default: val)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path for predictions TSV output')
    parser.add_argument('--report', type=str, default=None,
                       help='Path for metrics report (optional)')
    parser.add_argument('--ppm_tol', type=float, default=10.0,
                       help='Fragment matching tolerance in ppm (default: 10)')
    parser.add_argument('--min_intensity', type=float, default=0.01,
                       help='Minimum relative intensity threshold (default: 0.01)')
    parser.add_argument('--use_kmd', type=lambda x: x.lower() != 'false',
                       default=True,
                       help='Use Kendrick Mass Defect filtering for PFAS detection (default: True)')
    parser.add_argument('--kmd_threshold', type=float, default=0.15,
                       help='KMD threshold for PFAS classification (default: 0.15)')
    parser.add_argument('--use_mass_defect', type=lambda x: x.lower() != 'false',
                       default=True,
                       help='Use Mass Defect filtering for PFAS detection (default: True)')
    parser.add_argument('--mass_defect_min', type=float, default=0.9,
                       help='Minimum mass defect for PFAS classification (default: 0.9)')
    parser.add_argument('--mass_defect_max', type=float, default=0.99,
                       help='Maximum mass defect for PFAS classification (default: 0.99)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of validation samples to process (default: None = all)')

    args = parser.parse_args()

    print("=" * 80)
    print("PFAS Identification - MassQL Workflow")
    print("=" * 80)

    # Load validation fold (query spectra)
    df_val = load_data(args.input, args.fold)
    print(f"Loaded {len(df_val)} spectra from {args.fold} fold")

    # Sample validation data if requested
    if args.max_samples is not None and args.max_samples < len(df_val):
        print(f"Sampling {args.max_samples} validation spectra for faster processing...")
        df_val = df_val.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"Using {len(df_val)} validation samples\n")
    else:
        print()

    # Parse validation spectra
    print("Parsing validation spectra...")
    df_val['mzs_parsed'] = df_val['mzs'].apply(lambda x: [float(v) for v in x.split(',')])
    df_val['intensities_parsed'] = df_val['intensities'].apply(lambda x: [float(v) for v in x.split(',')])

    # Run MassQL PFAS detection
    print("\n" + "=" * 80)
    print("MassQL PFAS Detection")
    print("=" * 80)
    methods = ["CF2 loss detection", "diagnostic fragment matching"]
    if args.use_kmd:
        methods.append("Kendrick Mass Defect analysis")
    if args.use_mass_defect:
        methods.append(f"Mass Defect filtering ({args.mass_defect_min}-{args.mass_defect_max})")
    print(f"Running {', '.join(methods)}...\n")

    results = []
    for idx, row in df_val.iterrows():
        if idx % 100 == 0:
            print(f"  Processing spectrum {idx}/{len(df_val)}")

        result = compute_pfas_score(
            row,
            ppm_tol=args.ppm_tol,
            min_intensity=args.min_intensity,
            use_kmd=args.use_kmd,
            kmd_threshold=args.kmd_threshold,
            use_mass_defect=args.use_mass_defect,
            mass_defect_min=args.mass_defect_min,
            mass_defect_max=args.mass_defect_max
        )
        results.append(result)

    df_results = pd.concat([df_val, pd.DataFrame(results)], axis=1)

    # Save predictions
    print(f"\n" + "=" * 80)
    print(f"Saving predictions to {args.output}...")
    output_cols = ['identifier', 'is_PFAS', 'total_score', 'predicted_pfas',
                   'cf2_score', 'fragment_score', 'kmd_score', 'kmd', 'kendrick_mass',
                   'mass_defect_score', 'mass_defect', 'matched_fragments']
    df_results[output_cols].to_csv(args.output, sep='\t', index=False)
    print(f"Saved {len(df_results)} predictions")

    # Compute and display metrics
    print("\n" + "=" * 80)
    print("Classification Metrics")
    print("=" * 80)
    # Convert to boolean to handle mixed data types
    y_true = df_results['is_PFAS'].astype(bool)
    y_pred = df_results['predicted_pfas'].astype(bool)
    metrics = compute_metrics(y_true, y_pred)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    # Generate report
    if args.report:
        print(f"\nGenerating detailed report to {args.report}...")
        generate_report(df_results, metrics, args)
        print("Report generated successfully")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
