#!/usr/bin/env python3
"""
PFAS Identification CLI Tool

Implements the MassQL workflow for PFAS detection:
- Step 1: Molecular Networking (library-based spectral similarity)
- Step 2: MassQL Detection (CF2 losses + diagnostic fragments)

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


def load_external_library(filepath):
    """
    Load external spectral library (e.g., from MassBank).

    Expected TSV format:
    - identifier, compound_name, precursor_mz, mzs, intensities, is_PFAS, fold, source
    """
    print(f"Loading external library from {filepath}...")
    df = pd.read_csv(filepath, sep='\t')

    # Validate required columns
    required_cols = ['identifier', 'precursor_mz', 'mzs', 'intensities', 'is_PFAS']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"External library missing required columns: {missing_cols}")

    # Ensure is_PFAS is boolean
    df['is_PFAS'] = df['is_PFAS'].astype(bool)

    print(f"Loaded {len(df)} spectra from external library")
    if 'source' in df.columns:
        print(f"  Sources: {df['source'].value_counts().to_dict()}")

    return df


def build_hybrid_library(training_df, external_df, mode='hybrid'):
    """
    Build hybrid spectral library combining training and external data.

    Args:
        training_df: Training fold DataFrame
        external_df: External library DataFrame (e.g., MassBank)
        mode: 'hybrid' (both), 'external_only', or 'training_only'

    Returns:
        Combined DataFrame for molecular networking
    """
    if mode == 'external_only':
        print("Using EXTERNAL library only")
        library = external_df.copy()

    elif mode == 'training_only':
        print("Using TRAINING fold only")
        library = training_df.copy()

    else:  # hybrid
        print("Building HYBRID library (external + training)")

        # Prioritize external library (confirmed PFAS)
        external_pfas = external_df[external_df['is_PFAS'] == True].copy()

        # Add training data PFAS for dataset-specific patterns
        training_pfas = training_df[training_df['is_PFAS'] == True].copy()

        # Also include some non-PFAS from training for context
        training_non_pfas = training_df[training_df['is_PFAS'] == False].sample(
            n=min(1000, len(training_df[training_df['is_PFAS'] == False])),
            random_state=42
        )

        library = pd.concat([
            external_pfas,
            training_pfas,
            training_non_pfas
        ]).reset_index(drop=True)

        print(f"  - External PFAS: {len(external_pfas)}")
        print(f"  - Training PFAS: {len(training_pfas)}")
        print(f"  - Training non-PFAS (context): {len(training_non_pfas)}")
        print(f"  - Total library size: {len(library)}")

    return library


# =============================================================================
# Step 1: Molecular Networking Functions
# =============================================================================

def compute_cosine_similarity(mz1, int1, mz2, int2, mz_tolerance=0.01):
    """
    Calculate cosine similarity between two spectra.

    Matches peaks within mz_tolerance and computes:
    similarity = sum(intensity_i * intensity_j) / (norm1 * norm2)

    Args:
        mz1, int1: First spectrum (m/z and intensity lists)
        mz2, int2: Second spectrum
        mz_tolerance: Tolerance for peak matching (Daltons)

    Returns:
        Cosine similarity score [0, 1]
    """
    if len(int1) == 0 or len(int2) == 0:
        return 0.0

    # Normalize intensities
    int1_norm = np.array(int1) / np.linalg.norm(int1)
    int2_norm = np.array(int2) / np.linalg.norm(int2)

    # Match peaks and compute dot product
    dot_product = 0.0
    for i, mz_i in enumerate(mz1):
        for j, mz_j in enumerate(mz2):
            if abs(mz_i - mz_j) <= mz_tolerance:
                dot_product += int1_norm[i] * int2_norm[j]

    return dot_product


def query_against_library(query_df, library_df, similarity_threshold=0.7):
    """
    Query validation spectra against spectral library.

    For each query spectrum, find matching library spectra and propagate
    PFAS labels from library matches.

    Args:
        query_df: DataFrame with query spectra (validation fold)
        library_df: DataFrame with library spectra (training fold with is_PFAS labels)
        similarity_threshold: Minimum similarity for library match

    Returns:
        dict: {query_identifier: network_pfas_score}
    """
    network_scores = {}

    print(f"Querying {len(query_df)} validation spectra against {len(library_df)} library spectra...")

    for idx, query_row in query_df.iterrows():
        if idx % 50 == 0:
            print(f"  Querying spectrum {idx}/{len(query_df)}")

        query_id = query_row['identifier']
        query_mz = query_row['mzs_parsed']
        query_int = query_row['intensities_parsed']

        # Find library matches
        pfas_matches = 0
        total_matches = 0

        for _, lib_row in library_df.iterrows():
            sim = compute_cosine_similarity(
                query_mz, query_int,
                lib_row['mzs_parsed'],
                lib_row['intensities_parsed']
            )

            if sim >= similarity_threshold:
                total_matches += 1
                if lib_row['is_PFAS']:
                    pfas_matches += 1

        # Network score: proportion of PFAS matches in library
        network_scores[query_id] = pfas_matches / total_matches if total_matches > 0 else 0.0

    return network_scores


# =============================================================================
# Step 2: MassQL Detection Functions
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


def compute_pfas_score(row, ppm_tol=10, min_intensity=0.01, network_score=0.0, use_network=True, use_kmd=True, kmd_threshold=0.15):
    """
    Combine detection methods into overall PFAS score.

    Scoring:
    - CF2 losses: 2 points per detected CF2 unit
    - Diagnostic fragments: 3 points per matched fragment
    - Kendrick Mass Defect: 4 points if |KMD| <= threshold (default 0.15)
    - Network evidence: 5 points if network_score > 0.5 (connected to known PFAS)
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

    # Network evidence (if enabled)
    network_contribution = 0
    if use_network and network_score > 0.5:
        network_contribution = 5

    total_score = cf2_score + frag_score + kmd_score + network_contribution

    return {
        'cf2_score': cf2_score,
        'fragment_score': frag_score,
        'kmd_score': kmd_score,
        'kmd': kmd_value,
        'kendrick_mass': kendrick_mass,
        'network_score': network_score,
        'network_contribution': network_contribution,
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


def generate_report(df_results, metrics, args, network_stats):
    """Generate formatted text report."""
    with open(args.report, 'w') as f:
        f.write("PFAS Identification Results\n")
        f.write("============================\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Fold: {args.fold}\n")
        f.write(f"Method: MassQL with Molecular Networking\n")
        f.write(f"Samples: {len(df_results)}\n\n")

        f.write("Workflow:\n")
        f.write("Step 1: Molecular Networking (Library-Based)\n")
        if args.use_network_propagation:
            f.write(f"  - Spectral library: Training fold ({network_stats['library_size']} spectra)\n")
            f.write(f"  - Query spectra: Validation fold ({len(df_results)} spectra)\n")
            f.write(f"  - Similarity threshold: {args.network_similarity_threshold}\n")
            f.write(f"  - Validation spectra with library matches: {network_stats['queries_with_matches']}\n")
            f.write(f"  - Average library matches per query: {network_stats['avg_matches']:.2f}\n")
        else:
            f.write("  - Disabled\n")
        f.write("\n")

        f.write("Step 2: MassQL Detection\n")
        f.write("  - CF2 loss detection (perfluoroalkyl chains)\n")
        f.write("  - Diagnostic fragment matching\n")
        if args.use_kmd:
            f.write(f"  - Kendrick Mass Defect analysis (CF2 base, threshold={args.kmd_threshold})\n")
        else:
            f.write("  - Kendrick Mass Defect: Disabled\n")
        f.write("\n")

        f.write("Parameters:\n")
        f.write(f"- PPM tolerance: {args.ppm_tol}\n")
        f.write(f"- Minimum relative intensity: {args.min_intensity}\n")
        f.write("- CF2 score weight: 2\n")
        f.write("- Fragment score weight: 3\n")
        if args.use_kmd:
            f.write(f"- KMD score weight: 4 (if |KMD| <= {args.kmd_threshold})\n")
        f.write("- Network score weight: 5 (if network_score > 0.5)\n")
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

        # Network contribution
        if args.use_network_propagation:
            network_evidence = (df_results['network_contribution'] > 0).sum()
            network_only = ((df_results['cf2_score'] == 0) &
                          (df_results['fragment_score'] == 0) &
                          (df_results['predicted_pfas'] == True)).sum()
            f.write("Network Contribution:\n")
            f.write(f"- Samples with network evidence: {network_evidence}\n")
            f.write(f"- Network-only predictions (no CF2/fragments): {network_only}\n")


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
    parser.add_argument('--network_similarity_threshold', type=float, default=0.7,
                       help='Cosine similarity threshold for networking (default: 0.7)')
    parser.add_argument('--use_network_propagation', type=lambda x: x.lower() != 'false',
                       default=True,
                       help='Use molecular network to propagate PFAS labels (default: True)')
    parser.add_argument('--use_kmd', type=lambda x: x.lower() != 'false',
                       default=True,
                       help='Use Kendrick Mass Defect filtering for PFAS detection (default: True)')
    parser.add_argument('--kmd_threshold', type=float, default=0.15,
                       help='KMD threshold for PFAS classification (default: 0.15)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of validation samples to process (default: None = all)')
    parser.add_argument('--library_sample_size', type=int, default=None,
                       help='Number of training samples to use as library (default: None = all)')
    parser.add_argument('--external_library', type=str, default=None,
                       help='Path to external spectral library TSV (e.g., MassBank PFAS library)')
    parser.add_argument('--library_mode', type=str, default='hybrid',
                       choices=['hybrid', 'external_only', 'training_only'],
                       help='Library mode: hybrid (external+training), external_only, or training_only (default: hybrid)')

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

    # Step 1: Molecular Networking (if enabled)
    network_scores = {}
    network_stats = {'library_size': 0, 'queries_with_matches': 0, 'avg_matches': 0.0}

    if args.use_network_propagation:
        print("\n" + "=" * 80)
        print("Step 1: Molecular Networking")
        print("=" * 80)

        # Load external library if provided
        external_lib = None
        if args.external_library:
            external_lib = load_external_library(args.external_library)
            # Parse spectra
            external_lib['mzs_parsed'] = external_lib['mzs'].apply(lambda x: [float(v) for v in x.split(',')])
            external_lib['intensities_parsed'] = external_lib['intensities'].apply(lambda x: [float(v) for v in x.split(',')])

        # Load training fold
        df_train = load_data(args.input, 'train')
        print(f"Loaded {len(df_train)} training spectra")

        # Parse training spectra
        df_train['mzs_parsed'] = df_train['mzs'].apply(lambda x: [float(v) for v in x.split(',')])
        df_train['intensities_parsed'] = df_train['intensities'].apply(lambda x: [float(v) for v in x.split(',')])

        # Build library based on mode
        if external_lib is not None:
            library_df = build_hybrid_library(df_train, external_lib, mode=args.library_mode)
        else:
            print("Using TRAINING fold only (no external library)")
            library_df = df_train

        # Sample library if requested (after hybrid building)
        if args.library_sample_size is not None and args.library_sample_size < len(library_df):
            print(f"Sampling {args.library_sample_size} library spectra for faster processing...")
            library_df = library_df.sample(n=args.library_sample_size, random_state=42).reset_index(drop=True)

        print(f"Final library size: {len(library_df)} spectra\n")

        # Query validation spectra against library
        network_scores = query_against_library(
            df_val, library_df, args.network_similarity_threshold
        )

        # Calculate network stats
        network_stats['library_size'] = len(library_df)
        network_stats['queries_with_matches'] = sum(1 for s in network_scores.values() if s > 0)
        network_stats['avg_matches'] = np.mean([s for s in network_scores.values()])

        print(f"\nNetwork summary:")
        print(f"  - {network_stats['queries_with_matches']}/{len(df_val)} queries have library matches")
        print(f"  - Average PFAS score from network: {network_stats['avg_matches']:.3f}")
    else:
        print("\nMolecular networking disabled")
        network_scores = {row['identifier']: 0.0 for _, row in df_val.iterrows()}

    # Step 2: Run MassQL PFAS detection
    print("\n" + "=" * 80)
    print("Step 2: MassQL PFAS Detection")
    print("=" * 80)
    if args.use_kmd:
        print(f"Running CF2 loss detection, diagnostic fragment matching, and Kendrick Mass Defect analysis...\n")
    else:
        print("Running CF2 loss detection and diagnostic fragment matching...\n")

    results = []
    for idx, row in df_val.iterrows():
        if idx % 100 == 0:
            print(f"  Processing spectrum {idx}/{len(df_val)}")

        net_score = network_scores.get(row['identifier'], 0.0)
        result = compute_pfas_score(
            row,
            ppm_tol=args.ppm_tol,
            min_intensity=args.min_intensity,
            network_score=net_score,
            use_network=args.use_network_propagation,
            use_kmd=args.use_kmd,
            kmd_threshold=args.kmd_threshold
        )
        results.append(result)

    df_results = pd.concat([df_val, pd.DataFrame(results)], axis=1)

    # Save predictions
    print(f"\n" + "=" * 80)
    print(f"Saving predictions to {args.output}...")
    output_cols = ['identifier', 'is_PFAS', 'total_score', 'predicted_pfas',
                   'cf2_score', 'fragment_score', 'kmd_score', 'kmd', 'kendrick_mass',
                   'network_score', 'matched_fragments']
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
        generate_report(df_results, metrics, args, network_stats)
        print("Report generated successfully")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
