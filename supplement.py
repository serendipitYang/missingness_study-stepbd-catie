"""
Supplementary Materials Analysis
=================================
This module generates supplementary analyses including:
1. Population overlap reports (Venn diagrams)
2. KM curves for exclusive populations
3. Main effect analyses for smoking, BMI, and illness severity
4. Multivariate Cox regression models
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings
from matplotlib_venn import venn2
from lifelines import KaplanMeierFitter, CoxPHFitter
from statsmodels.stats.multitest import fdrcorrection
from utils import (
    prepare_data_for_visualization,
    format_table_for_excel,
    save_table_to_excel,
    _extract_row_from_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# Helper Functions
# =============================================================================

def clean_survival_data(df, required_columns=['daysrz', 'event_occurs']):
    """
    Clean and validate survival data.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    required_columns : list
        Columns that must be present and numeric

    Returns:
    --------
    DataFrame : Cleaned data
    """
    df = df.copy()

    # Ensure required columns are numeric
    for col in required_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)

    return df


# =============================================================================
# 1. Population Overlap Analysis (Venn Diagrams)
# =============================================================================

def plot_population_overlap(smoking_df, bmi_df, dataset_name, ax=None):
    """
    Create Venn diagram showing population overlap between smoking and BMI studies.

    Note: Applies the same exclusion criteria used in the main analyses:
    - Removes subjects with missing baseline data
    - Removes underweight subjects (bmi_cat == 1) from BMI study

    Parameters:
    -----------
    smoking_df : DataFrame
        Smoking study population
    bmi_df : DataFrame
        BMI study population
    dataset_name : str
        Name of the dataset (StepBD or CATIE)
    ax : matplotlib axis, optional
        Axis to plot on

    Returns:
    --------
    dict : Population counts
    """
    # Filter smoking dataset: remove missing baseline
    smoking_df_filtered = smoking_df.copy()

    # Remove subjects with missing baseline data in smoking dataset
    # Use smoker_yn_1 column to exclude 'missing_baseliner'
    if 'smoker_yn_1' in smoking_df_filtered.columns:
        smoking_df_filtered = smoking_df_filtered[smoking_df_filtered['smoker_yn_1'] != 'missing_baseliner']

    # Filter BMI dataset: remove missing baseline AND underweight
    bmi_df_filtered = bmi_df.copy()

    # Remove subjects with no_bmi (missing baseline) and bmi1 (underweight)
    if 'bmi_status' in bmi_df_filtered.columns:
        bmi_df_filtered = bmi_df_filtered[~bmi_df_filtered['bmi_status'].isin(['no_bmi', 'bmi1'])]
    elif 'bmi_cat' in bmi_df_filtered.columns:
        # Fallback: use bmi_cat if bmi_status is not available
        bmi_df_filtered = bmi_df_filtered[bmi_df_filtered['bmi_cat'].isin([2, 3, 4])]

    # Get unique subjects after filtering
    smoking_subjects = set(smoking_df_filtered['subjectkey'].unique())
    bmi_subjects = set(bmi_df_filtered['subjectkey'].unique())

    only_smoking = len(smoking_subjects - bmi_subjects)
    only_bmi = len(bmi_subjects - smoking_subjects)
    both = len(smoking_subjects & bmi_subjects)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    venn2(subsets=(only_smoking, only_bmi, both),
          set_labels=('Smoking Study', 'BMI Study'),
          ax=ax)

    ax.set_title(f'{dataset_name} Population Overlap')

    logger.info(f"{dataset_name} - After exclusions:")
    logger.info(f"  Total in Smoking Study: {len(smoking_subjects)}")
    logger.info(f"  Total in BMI Study: {len(bmi_subjects)}")
    logger.info(f"  Only in Smoking: {only_smoking}")
    logger.info(f"  Only in BMI: {only_bmi}")
    logger.info(f"  In Both: {both}")

    return {
        'only_smoking': only_smoking,
        'only_bmi': only_bmi,
        'both': both,
        'total_smoking': len(smoking_subjects),
        'total_bmi': len(bmi_subjects)
    }


def create_population_overlap_report(df_stepbd_smoking, df_catie_smoking,
                                     df_stepbd_bmi, df_catie_bmi,
                                     output_dir=None):
    """
    Create comprehensive population overlap report with Venn diagrams.

    Parameters:
    -----------
    df_stepbd_smoking, df_catie_smoking : DataFrames
        Smoking study populations
    df_stepbd_bmi, df_catie_bmi : DataFrames
        BMI study populations
    output_dir : str, optional
        Directory to save figures

    Returns:
    --------
    DataFrame : Summary statistics
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # StepBD overlap
    stepbd_counts = plot_population_overlap(
        df_stepbd_smoking, df_stepbd_bmi, 'StepBD', axes[0]
    )

    # CATIE overlap
    catie_counts = plot_population_overlap(
        df_catie_smoking, df_catie_bmi, 'CATIE', axes[1]
    )

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'Supplement_Figure_1_Population_Overlap.png'),
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Create summary table
    summary = pd.DataFrame([
        {
            'Dataset': 'StepBD',
            'Total in Smoking Study': stepbd_counts['total_smoking'],
            'Total in BMI Study': stepbd_counts['total_bmi'],
            'Only in Smoking Study': stepbd_counts['only_smoking'],
            'Only in BMI Study': stepbd_counts['only_bmi'],
            'In Both Studies': stepbd_counts['both']
        },
        {
            'Dataset': 'CATIE',
            'Total in Smoking Study': catie_counts['total_smoking'],
            'Total in BMI Study': catie_counts['total_bmi'],
            'Only in Smoking Study': catie_counts['only_smoking'],
            'Only in BMI Study': catie_counts['only_bmi'],
            'In Both Studies': catie_counts['both']
        }
    ])

    if output_dir:
        summary.to_csv(os.path.join(output_dir, 'Supplement_Table_1_Population_Overlap.csv'),
                      index=False)

    logger.info("Population overlap report created")
    return summary


# =============================================================================
# 2. KM Curves for Exclusive Populations
# =============================================================================

def create_exclusive_population_datasets(smoking_df, bmi_df):
    """
    Create datasets for subjects only in smoking or only in BMI studies.

    IMPORTANT: Applies the same exclusion criteria as the main analyses:
    - Removes subjects with missing baseline from smoking study
    - Removes subjects with missing baseline and underweight from BMI study

    Parameters:
    -----------
    smoking_df, bmi_df : DataFrames
        Input datasets

    Returns:
    --------
    tuple : (only_smoking_df, only_bmi_df)
    """
    # Apply exclusion criteria FIRST (same as in plot_population_overlap)
    smoking_df_filtered = smoking_df.copy()
    if 'smoker_yn_1' in smoking_df_filtered.columns:
        smoking_df_filtered = smoking_df_filtered[smoking_df_filtered['smoker_yn_1'] != 'missing_baseliner']

    bmi_df_filtered = bmi_df.copy()
    if 'bmi_status' in bmi_df_filtered.columns:
        bmi_df_filtered = bmi_df_filtered[~bmi_df_filtered['bmi_status'].isin(['no_bmi', 'bmi1'])]
    elif 'bmi_cat' in bmi_df_filtered.columns:
        bmi_df_filtered = bmi_df_filtered[bmi_df_filtered['bmi_cat'].isin([2, 3, 4])]

    # Now get unique subjects from filtered datasets
    smoking_subjects = set(smoking_df_filtered['subjectkey'].unique())
    bmi_subjects = set(bmi_df_filtered['subjectkey'].unique())

    only_smoking_subjects = smoking_subjects - bmi_subjects
    only_bmi_subjects = bmi_subjects - smoking_subjects

    # Create datasets with only exclusive subjects
    only_smoking_df = smoking_df_filtered[smoking_df_filtered['subjectkey'].isin(only_smoking_subjects)].copy()
    only_bmi_df = bmi_df_filtered[bmi_df_filtered['subjectkey'].isin(only_bmi_subjects)].copy()

    logger.info(f"Exclusive populations created:")
    logger.info(f"  Only in Smoking: {len(only_smoking_subjects)} subjects, {len(only_smoking_df)} observations")
    logger.info(f"  Only in BMI: {len(only_bmi_subjects)} subjects, {len(only_bmi_df)} observations")

    return only_smoking_df, only_bmi_df


def plot_km_exclusive_populations(df_stepbd_smoking, df_catie_smoking,
                                  df_stepbd_bmi, df_catie_bmi,
                                  output_dir=None):
    """
    Plot KM curves comparing exclusive populations.

    Parameters:
    -----------
    df_stepbd_smoking, df_catie_smoking : DataFrames
        Smoking datasets
    df_stepbd_bmi, df_catie_bmi : DataFrames
        BMI datasets
    output_dir : str, optional
        Directory to save figure

    Returns:
    --------
    dict : Median survival times
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    median_times = {}

    for idx, (smoking_df, bmi_df, dataset_name) in enumerate([
        (df_stepbd_smoking, df_stepbd_bmi, 'StepBD'),
        (df_catie_smoking, df_catie_bmi, 'CATIE')
    ]):
        ax = axes[idx]

        # Create exclusive populations (already applies exclusion criteria)
        only_smoking_df, only_bmi_df = create_exclusive_population_datasets(
            smoking_df, bmi_df
        )

        # Clean survival data (remove rows with missing daysrz/event_occurs)
        only_smoking_df = clean_survival_data(only_smoking_df)
        only_bmi_df = clean_survival_data(only_bmi_df)

        # Check if we have enough data for KM curves
        min_observations = 10  # Minimum observations needed for meaningful KM curve
        if len(only_smoking_df) < min_observations and len(only_bmi_df) < min_observations:
            logger.warning(f"{dataset_name}: Insufficient data in exclusive populations")
            logger.warning(f"  Only Smoking: {len(only_smoking_df)} observations")
            logger.warning(f"  Only BMI: {len(only_bmi_df)} observations")
            ax.text(0.5, 0.5, f'Insufficient data for {dataset_name}\n' +
                   f'Only Smoking: {len(only_smoking_df)} obs\n' +
                   f'Only BMI: {len(only_bmi_df)} obs',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{dataset_name}')
            median_times[dataset_name] = {
                'Only in Smoking Study': np.nan,
                'Only in BMI Study': np.nan
            }
            continue

        # Fit and plot KM models (only if sufficient data)
        median_smoking = np.nan
        median_bmi = np.nan

        if len(only_smoking_df) >= min_observations:
            only_smoking_df['weight'] = 1 / only_smoking_df.groupby('subjectkey')['subjectkey'].transform('count')
            kmf_smoking = KaplanMeierFitter()
            kmf_smoking.fit(
                only_smoking_df['daysrz'],
                event_observed=only_smoking_df['event_occurs'],
                weights=only_smoking_df['weight'],
                label='Only in Smoking Study'
            )
            kmf_smoking.plot(ax=ax, ci_alpha=0)
            median_smoking = kmf_smoking.median_survival_time_
            logger.info(f"{dataset_name} - Only Smoking median: {median_smoking:.2f} days")
        else:
            logger.warning(f"{dataset_name} - Only Smoking: {len(only_smoking_df)} observations (< {min_observations}), skipping")

        if len(only_bmi_df) >= min_observations:
            only_bmi_df['weight'] = 1 / only_bmi_df.groupby('subjectkey')['subjectkey'].transform('count')
            kmf_bmi = KaplanMeierFitter()
            kmf_bmi.fit(
                only_bmi_df['daysrz'],
                event_observed=only_bmi_df['event_occurs'],
                weights=only_bmi_df['weight'],
                label='Only in BMI Study'
            )
            kmf_bmi.plot(ax=ax, ci_alpha=0)
            median_bmi = kmf_bmi.median_survival_time_
            logger.info(f"{dataset_name} - Only BMI median: {median_bmi:.2f} days")
        else:
            logger.warning(f"{dataset_name} - Only BMI: {len(only_bmi_df)} observations (< {min_observations}), skipping")

        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')

        if dataset_name == 'CATIE':
            ax.set_xlim(0, 550)
            ax.set_ylim(0, 1)

        median_times[dataset_name] = {
            'Only in Smoking Study': median_smoking,
            'Only in BMI Study': median_bmi
        }

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'Supplement_Figure_2_KM_Exclusive_Populations.png'),
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return median_times


# =============================================================================
# 3. Main Effect: Smoking (PWS vs PWDS) - Whole Set Analysis
# =============================================================================

def analyze_smoking_main_effect(df, dataset_name, with_weights=True):
    """
    Analyze main effect of smoking on entire population (not pairwise).

    Parameters:
    -----------
    df : DataFrame
        Input dataset with smoking data
    dataset_name : str
        Dataset name
    with_weights : bool
        Whether to use weights

    Returns:
    --------
    tuple : (KaplanMeierFitter objects, CoxPHFitter model, stats dict)
    """
    df = df.copy()

    # Clean data first
    df = clean_survival_data(df)

    # Filter to smokers (2) and non-smokers (0), exclude changers (1)
    df_filtered = df[df['smoker_yn'].isin([0, 2])].copy()
    df_filtered['is_smoker'] = (df_filtered['smoker_yn'] == 2).astype(int)
    df_filtered['is_smoker_cat'] = pd.Categorical(df_filtered['is_smoker'],
                                                    categories=[0, 1], ordered=True)

    # Add weights if needed
    if with_weights:
        df_filtered['weight'] = 1 / df_filtered.groupby('subjectkey')['subjectkey'].transform('count')
        weights_col = 'weight'
    else:
        weights_col = None

    # Fit KM curves
    kmf_pwds = KaplanMeierFitter()
    kmf_pws = KaplanMeierFitter()

    pwds_data = df_filtered[df_filtered['is_smoker'] == 0]
    pws_data = df_filtered[df_filtered['is_smoker'] == 1]

    kmf_pwds.fit(
        pwds_data['daysrz'],
        event_observed=pwds_data['event_occurs'],
        weights=pwds_data[weights_col] if weights_col else None,
        label='PWDS (Non-smokers)'
    )

    kmf_pws.fit(
        pws_data['daysrz'],
        event_observed=pws_data['event_occurs'],
        weights=pws_data[weights_col] if weights_col else None,
        label='PWS (Smokers)'
    )

    # Fit Cox model
    cph = CoxPHFitter()
    model_params = {
        'duration_col': 'daysrz',
        'event_col': 'event_occurs',
        'formula': 'is_smoker_cat'
    }
    if weights_col:
        model_params['weights_col'] = weights_col

    cph.fit(df_filtered[['daysrz', 'event_occurs', 'is_smoker_cat'] +
                        ([weights_col] if weights_col else [])], **model_params)

    # Extract statistics
    stats = _extract_row_from_model(cph, 'is_smoker_cat')
    stats['Number of Observations'] = len(df_filtered[df_filtered['is_smoker'] == 0]) if not with_weights else len(df_filtered)
    stats['Dataset'] = dataset_name
    stats['Comparison'] = 'PWS vs. PWDS'

    return kmf_pwds, kmf_pws, cph, stats


def plot_smoking_main_effect(df_stepbd, df_catie, output_dir=None):
    """
    Plot main effect KM curves for smoking.

    Returns:
    --------
    DataFrame : Cox regression results
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    results = []

    for idx, (df, dataset_name) in enumerate([(df_stepbd, 'StepBD'), (df_catie, 'CATIE')]):
        ax = axes[idx]
        kmf_pwds, kmf_pws, cph, stats = analyze_smoking_main_effect(df, dataset_name)

        # Plot
        kmf_pwds.plot(ax=ax, ci_alpha=0)
        kmf_pws.plot(ax=ax, ci_alpha=0)

        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')

        if dataset_name == 'CATIE':
            ax.set_xlim(0, 550)

        results.append(stats)

        logger.info(f"{dataset_name} - PWDS median: {kmf_pwds.median_survival_time_:.2f} days")
        logger.info(f"{dataset_name} - PWS median: {kmf_pws.median_survival_time_:.2f} days")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'Supplement_Figure_3_Smoking_Main_Effect.png'),
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Create results table
    results_df = pd.DataFrame(results)

    # FDR correction
    _, p_adj = fdrcorrection([r['Uncorrected p'] for r in results], alpha=0.05)
    results_df['FDR-corrected p'] = p_adj

    # Reorder columns
    results_df = results_df[[
        'Dataset', 'Comparison', 'Number of Observations',
        'HR', 'HR, lower 95%', 'HR, upper 95%', 'z', 'Uncorrected p', 'FDR-corrected p'
    ]]

    if output_dir:
        results_df.to_csv(os.path.join(output_dir, 'Supplement_Table_2_Smoking_Main_Effect.csv'),
                         index=False)
        save_table_to_excel(results_df,
                          os.path.join(output_dir, 'Supplement_Table_2_Smoking_Main_Effect.xlsx'),
                          sheet_name='Smoking Main Effect')

    return results_df


# =============================================================================
# 4. Main Effect: BMI
# =============================================================================

def analyze_bmi_main_effect(df, dataset_name, with_weights=True):
    """
    Analyze main effect of BMI categories.

    Returns:
    --------
    tuple : (KM fitters dict, Cox models dict, stats list)
    """
    df = df.copy()

    # Clean data first
    df = clean_survival_data(df)

    # Filter to valid BMI categories (2=Normal, 3=Overweight, 4=Obese)
    df_filtered = df[df['bmi_cat'].isin([2, 3, 4])].copy()

    # Add weights
    if with_weights:
        df_filtered['weight'] = 1 / df_filtered.groupby('subjectkey')['subjectkey'].transform('count')
        weights_col = 'weight'
    else:
        weights_col = None

    # Fit KM curves for each BMI category
    kmf_normal = KaplanMeierFitter()
    kmf_overweight = KaplanMeierFitter()
    kmf_obese = KaplanMeierFitter()

    normal_data = df_filtered[df_filtered['bmi_cat'] == 2]
    overweight_data = df_filtered[df_filtered['bmi_cat'] == 3]
    obese_data = df_filtered[df_filtered['bmi_cat'] == 4]

    kmf_normal.fit(
        normal_data['daysrz'],
        event_observed=normal_data['event_occurs'],
        weights=normal_data[weights_col] if weights_col else None,
        label='Normal Weight'
    )

    kmf_overweight.fit(
        overweight_data['daysrz'],
        event_observed=overweight_data['event_occurs'],
        weights=overweight_data[weights_col] if weights_col else None,
        label='Overweight'
    )

    kmf_obese.fit(
        obese_data['daysrz'],
        event_observed=obese_data['event_occurs'],
        weights=obese_data[weights_col] if weights_col else None,
        label='Obese'
    )

    # Cox models for each comparison
    stats = []

    # Overweight vs Normal
    df_ow_vs_n = df_filtered[df_filtered['bmi_cat'].isin([2, 3])].copy()
    df_ow_vs_n['is_overweight'] = (df_ow_vs_n['bmi_cat'] == 3).astype(int)
    df_ow_vs_n['is_overweight_cat'] = pd.Categorical(df_ow_vs_n['is_overweight'],
                                                       categories=[0, 1], ordered=True)

    cph_ow = CoxPHFitter()
    model_params = {
        'duration_col': 'daysrz',
        'event_col': 'event_occurs',
        'formula': 'is_overweight_cat'
    }
    if weights_col:
        model_params['weights_col'] = weights_col

    cph_ow.fit(df_ow_vs_n[['daysrz', 'event_occurs', 'is_overweight_cat'] +
                          ([weights_col] if weights_col else [])], **model_params)

    stats_ow = _extract_row_from_model(cph_ow, 'is_overweight_cat')
    stats_ow['Number of Observations'] = len(df_ow_vs_n)
    stats_ow['Dataset'] = dataset_name
    stats_ow['Comparison'] = 'Overweight vs. Normal weight'
    stats.append(stats_ow)

    # Obese vs Normal
    df_ob_vs_n = df_filtered[df_filtered['bmi_cat'].isin([2, 4])].copy()
    df_ob_vs_n['is_obese'] = (df_ob_vs_n['bmi_cat'] == 4).astype(int)
    df_ob_vs_n['is_obese_cat'] = pd.Categorical(df_ob_vs_n['is_obese'],
                                                  categories=[0, 1], ordered=True)

    cph_ob = CoxPHFitter()
    model_params['formula'] = 'is_obese_cat'

    cph_ob.fit(df_ob_vs_n[['daysrz', 'event_occurs', 'is_obese_cat'] +
                          ([weights_col] if weights_col else [])], **model_params)

    stats_ob = _extract_row_from_model(cph_ob, 'is_obese_cat')
    stats_ob['Number of Observations'] = len(df_ob_vs_n)
    stats_ob['Dataset'] = dataset_name
    stats_ob['Comparison'] = 'Obese vs. Normal weight'
    stats.append(stats_ob)

    return {
        'normal': kmf_normal,
        'overweight': kmf_overweight,
        'obese': kmf_obese
    }, stats


def plot_bmi_main_effect(df_stepbd, df_catie, output_dir=None):
    """
    Plot main effect KM curves for BMI.

    Returns:
    --------
    DataFrame : Cox regression results
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    all_stats = []

    for idx, (df, dataset_name) in enumerate([(df_stepbd, 'StepBD'), (df_catie, 'CATIE')]):
        ax = axes[idx]
        kmf_dict, stats = analyze_bmi_main_effect(df, dataset_name)

        # Plot
        kmf_dict['normal'].plot(ax=ax, ci_alpha=0)
        kmf_dict['overweight'].plot(ax=ax, ci_alpha=0)
        kmf_dict['obese'].plot(ax=ax, ci_alpha=0)

        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')

        if dataset_name == 'CATIE':
            ax.set_xlim(0, 550)

        all_stats.extend(stats)

        logger.info(f"{dataset_name} - Normal median: {kmf_dict['normal'].median_survival_time_:.2f} days")
        logger.info(f"{dataset_name} - Overweight median: {kmf_dict['overweight'].median_survival_time_:.2f} days")
        logger.info(f"{dataset_name} - Obese median: {kmf_dict['obese'].median_survival_time_:.2f} days")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'Supplement_Figure_4_BMI_Main_Effect.png'),
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Create results table
    results_df = pd.DataFrame(all_stats)

    # FDR correction
    _, p_adj = fdrcorrection([r['Uncorrected p'] for r in all_stats], alpha=0.05)
    results_df['FDR-corrected p'] = p_adj

    # Reorder columns
    results_df = results_df[[
        'Dataset', 'Comparison', 'Number of Observations',
        'HR', 'HR, lower 95%', 'HR, upper 95%', 'z', 'Uncorrected p', 'FDR-corrected p'
    ]]

    if output_dir:
        results_df.to_csv(os.path.join(output_dir, 'Supplement_Table_3_BMI_Main_Effect.csv'),
                         index=False)
        save_table_to_excel(results_df,
                          os.path.join(output_dir, 'Supplement_Table_3_BMI_Main_Effect.xlsx'),
                          sheet_name='BMI Main Effect')

    return results_df


# =============================================================================
# 5. Main Effect: Illness Severity
# =============================================================================

def analyze_illness_severity_main_effect(df, dataset_name, with_weights=True):
    """
    Analyze main effect of illness severity (unwell vs well).

    Returns:
    --------
    tuple : (KM fitters, Cox model, stats dict)
    """
    df = df.copy()

    # Clean data first
    df = clean_survival_data(df)

    # Filter to valid clinical status
    df_filtered = df[df['clinstat_2'].isin([0, 1])].copy()
    df_filtered['is_unwell'] = df_filtered['clinstat_2'].astype(int)
    df_filtered['is_unwell_cat'] = pd.Categorical(df_filtered['is_unwell'],
                                                    categories=[0, 1], ordered=True)

    # Add weights
    if with_weights:
        df_filtered['weight'] = 1 / df_filtered.groupby('subjectkey')['subjectkey'].transform('count')
        weights_col = 'weight'
    else:
        weights_col = None

    # Fit KM curves
    kmf_well = KaplanMeierFitter()
    kmf_unwell = KaplanMeierFitter()

    well_data = df_filtered[df_filtered['is_unwell'] == 0]
    unwell_data = df_filtered[df_filtered['is_unwell'] == 1]

    kmf_well.fit(
        well_data['daysrz'],
        event_observed=well_data['event_occurs'],
        weights=well_data[weights_col] if weights_col else None,
        label='Well'
    )

    kmf_unwell.fit(
        unwell_data['daysrz'],
        event_observed=unwell_data['event_occurs'],
        weights=unwell_data[weights_col] if weights_col else None,
        label='Unwell'
    )

    # Fit Cox model
    cph = CoxPHFitter()
    model_params = {
        'duration_col': 'daysrz',
        'event_col': 'event_occurs',
        'formula': 'is_unwell_cat'
    }
    if weights_col:
        model_params['weights_col'] = weights_col

    cph.fit(df_filtered[['daysrz', 'event_occurs', 'is_unwell_cat'] +
                        ([weights_col] if weights_col else [])], **model_params)

    # Extract statistics
    stats = _extract_row_from_model(cph, 'is_unwell_cat')
    stats['Number of Observations'] = len(df_filtered)
    stats['Dataset'] = dataset_name
    stats['Comparison'] = 'Unwell vs. Well'

    return kmf_well, kmf_unwell, cph, stats


def plot_illness_severity_main_effect(df_stepbd, df_catie, output_dir=None):
    """
    Plot main effect KM curves for illness severity.

    Returns:
    --------
    DataFrame : Cox regression results
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    results = []

    for idx, (df, dataset_name) in enumerate([(df_stepbd, 'StepBD'), (df_catie, 'CATIE')]):
        ax = axes[idx]
        kmf_well, kmf_unwell, cph, stats = analyze_illness_severity_main_effect(df, dataset_name)

        # Plot
        kmf_well.plot(ax=ax, ci_alpha=0)
        kmf_unwell.plot(ax=ax, ci_alpha=0)

        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')

        if dataset_name == 'CATIE':
            ax.set_xlim(0, 550)

        results.append(stats)

        logger.info(f"{dataset_name} - Well median: {kmf_well.median_survival_time_:.2f} days")
        logger.info(f"{dataset_name} - Unwell median: {kmf_unwell.median_survival_time_:.2f} days")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'Supplement_Figure_5_Illness_Severity_Main_Effect.png'),
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Create results table
    results_df = pd.DataFrame(results)

    # FDR correction
    _, p_adj = fdrcorrection([r['Uncorrected p'] for r in results], alpha=0.05)
    results_df['FDR-corrected p'] = p_adj

    # Reorder columns
    results_df = results_df[[
        'Dataset', 'Comparison', 'Number of Observations',
        'HR', 'HR, lower 95%', 'HR, upper 95%', 'z', 'Uncorrected p', 'FDR-corrected p'
    ]]

    if output_dir:
        results_df.to_csv(os.path.join(output_dir, 'Supplement_Table_4_Illness_Severity_Main_Effect.csv'),
                         index=False)
        save_table_to_excel(results_df,
                          os.path.join(output_dir, 'Supplement_Table_4_Illness_Severity_Main_Effect.xlsx'),
                          sheet_name='Illness Severity')

    return results_df


# =============================================================================
# 6. Multivariate Cox: Smoking + Illness Severity
# =============================================================================

def multivariate_smoking_illness(df, dataset_name, with_weights=True):
    """
    Multivariate Cox regression: Smoking + Illness Severity.

    Same comparisons as Table 2 but with multivariate adjustment:
    1. PWS vs. well PWDS (adjusted for illness severity)
    2. Unwell vs. well PWDS (adjusted for smoking status)
    3. Unwell PWS vs. well PWDS (adjusted)

    Returns:
    --------
    list : Statistics for all 3 comparisons
    """
    df = df.copy()

    # Clean data first
    df = clean_survival_data(df)

    # Set up categorical variables
    df['smoker_yn_c'] = pd.Categorical(df['smoker_yn'], categories=[0, 1, 2], ordered=True)
    df['clinstat_2_c'] = pd.Categorical(df['clinstat_2'], categories=[0, 1], ordered=True)

    # Add weights
    if with_weights:
        df['weight'] = 1 / df.groupby('subjectkey')['subjectkey'].transform('count')
        weights_col = 'weight'
    else:
        weights_col = None

    results = []

    # 1. PWS vs. well PWDS, adjusted for illness severity
    df_comp = df[
        ((df['smoker_yn'] == 2)) |  # Smoker
        ((df['smoker_yn'] == 0) & (df['clinstat_2'] == 0))  # Well non-smoker
    ].copy()
    df_comp['comparison'] = df_comp.apply(
        lambda row: 1 if row['smoker_yn'] == 2 else 0, axis=1
    )
    df_comp['comparison_cat'] = pd.Categorical(df_comp['comparison'], categories=[0, 1], ordered=True)
    df_comp['clinstat_2_c'] = pd.Categorical(df_comp['clinstat_2'], categories=[0, 1], ordered=True)

    cph = CoxPHFitter()
    model_params = {
        'duration_col': 'daysrz',
        'event_col': 'event_occurs',
        'formula': 'comparison_cat + clinstat_2_c'
    }
    if weights_col:
        model_params['weights_col'] = weights_col

    cph.fit(df_comp[['daysrz', 'event_occurs', 'comparison_cat', 'clinstat_2_c'] +
                    ([weights_col] if weights_col else [])], **model_params)

    stats = _extract_row_from_model(cph, 'comparison_cat')
    stats['Comparison'] = 'PWS vs. well PWDS'
    stats['Dataset'] = dataset_name
    stats['Number of Observations'] = len(df_comp)
    results.append(stats)

    # 2. Unwell vs. well PWDS, adjusted for smoking status
    df_comp = df[
        (df['clinstat_2'] == 1) |  # Unwell
        ((df['smoker_yn'] == 0) & (df['clinstat_2'] == 0))  # Well non-smoker
    ].copy()
    df_comp['comparison'] = df_comp.apply(
        lambda row: 1 if row['clinstat_2'] == 1 else 0, axis=1
    )
    df_comp['comparison_cat'] = pd.Categorical(df_comp['comparison'], categories=[0, 1], ordered=True)
    df_comp['smoker_yn_binary'] = (df_comp['smoker_yn'] == 2).astype(int)
    df_comp['smoker_yn_binary_cat'] = pd.Categorical(df_comp['smoker_yn_binary'], categories=[0, 1], ordered=True)

    cph = CoxPHFitter()
    model_params['formula'] = 'comparison_cat + smoker_yn_binary_cat'

    cph.fit(df_comp[['daysrz', 'event_occurs', 'comparison_cat', 'smoker_yn_binary_cat'] +
                    ([weights_col] if weights_col else [])], **model_params)

    stats = _extract_row_from_model(cph, 'comparison_cat')
    stats['Comparison'] = 'Unwell vs. well PWDS'
    stats['Dataset'] = dataset_name
    stats['Number of Observations'] = len(df_comp)
    results.append(stats)

    # 3. Unwell PWS vs. well PWDS
    # Note: Cannot use interaction term here due to perfect collinearity
    # (filtering to only 2 groups makes interaction = comparison variable)
    # Use simple univariate comparison instead
    df_comp = df[
        ((df['smoker_yn'] == 2) & (df['clinstat_2'] == 1)) |  # Unwell smoker
        ((df['smoker_yn'] == 0) & (df['clinstat_2'] == 0))  # Well non-smoker
    ].copy()
    df_comp['comparison'] = df_comp.apply(
        lambda row: 1 if (row['smoker_yn'] == 2 and row['clinstat_2'] == 1) else 0, axis=1
    )
    df_comp['comparison_cat'] = pd.Categorical(df_comp['comparison'], categories=[0, 1], ordered=True)

    cph = CoxPHFitter()
    model_params['formula'] = 'comparison_cat'

    cph.fit(df_comp[['daysrz', 'event_occurs', 'comparison_cat'] +
                    ([weights_col] if weights_col else [])], **model_params)

    stats = _extract_row_from_model(cph, 'comparison_cat')
    stats['Comparison'] = 'Unwell PWS vs. well PWDS'
    stats['Dataset'] = dataset_name
    stats['Number of Observations'] = len(df_comp)
    results.append(stats)

    return results


def create_multivariate_smoking_illness_table(df_stepbd, df_catie, output_dir=None):
    """
    Create multivariate Cox regression table for Smoking + Illness Severity.

    Same comparisons as Table 2, but with multivariate adjustment:
    - 3 rows per dataset (6 total)

    Returns:
    --------
    DataFrame : Results table
    """
    results = []

    # StepBD
    stepbd_stats = multivariate_smoking_illness(df_stepbd, 'StepBD')
    results.extend(stepbd_stats)

    # CATIE
    catie_stats = multivariate_smoking_illness(df_catie, 'CATIE')
    results.extend(catie_stats)

    # Create table
    results_df = pd.DataFrame(results)

    # FDR correction across all 6 rows
    _, p_adj = fdrcorrection([r['Uncorrected p'] for r in results], alpha=0.05)
    results_df['FDR-corrected p'] = p_adj

    # Reorder columns
    results_df = results_df[[
        'Dataset', 'Comparison', 'Number of Observations',
        'HR', 'HR, lower 95%', 'HR, upper 95%', 'z', 'Uncorrected p', 'FDR-corrected p'
    ]]

    # Sort by dataset and comparison order
    comparison_order = {
        'PWS vs. well PWDS': 0,
        'Unwell vs. well PWDS': 1,
        'Unwell PWS vs. well PWDS': 2
    }
    results_df['order'] = results_df['Dataset'].map({'StepBD': 0, 'CATIE': 1}) + \
                          results_df['Comparison'].map(comparison_order) / 10.0
    results_df = results_df.sort_values('order').drop(columns=['order']).reset_index(drop=True)

    if output_dir:
        results_df.to_csv(os.path.join(output_dir,
                         'Supplement_Table_5_Multivariate_Smoking_Illness.csv'),
                         index=False)
        save_table_to_excel(results_df,
                          os.path.join(output_dir,
                                     'Supplement_Table_5_Multivariate_Smoking_Illness.xlsx'),
                          sheet_name='Multivariate')

    logger.info("Multivariate Smoking + Illness Severity table created")
    return results_df


# =============================================================================
# 7. Multivariate Cox: BMI + Illness Severity
# =============================================================================

def multivariate_bmi_illness(df, dataset_name, with_weights=True):
    """
    Multivariate Cox regression: BMI + Illness Severity.

    Same comparisons as Table 3 but with multivariate adjustment:
    1. Overweight vs. well normal weight (adjusted for illness)
    2. Obese vs. well normal weight (adjusted for illness)
    3. Unwell vs. well normal weight (adjusted for BMI)
    4. Unwell overweight vs. well normal weight
    5. Unwell obese vs. well normal weight

    Returns:
    --------
    list : Statistics for all 5 comparisons
    """
    df = df.copy()

    # Clean data first
    df = clean_survival_data(df)

    # Set up categorical variables
    df['bmi_cat_c'] = pd.Categorical(df['bmi_cat'], categories=[2, 3, 4], ordered=True)
    df['clinstat_2_c'] = pd.Categorical(df['clinstat_2'], categories=[0, 1], ordered=True)

    # Add weights
    if with_weights:
        df['weight'] = 1 / df.groupby('subjectkey')['subjectkey'].transform('count')
        weights_col = 'weight'
    else:
        weights_col = None

    results = []

    # 1. Overweight vs. well normal weight, adjusted for illness severity
    df_comp = df[
        (df['bmi_cat'] == 3) |  # Overweight
        ((df['bmi_cat'] == 2) & (df['clinstat_2'] == 0))  # Well normal weight
    ].copy()
    df_comp['comparison'] = df_comp.apply(
        lambda row: 1 if row['bmi_cat'] == 3 else 0, axis=1
    )
    df_comp['comparison_cat'] = pd.Categorical(df_comp['comparison'], categories=[0, 1], ordered=True)
    df_comp['clinstat_2_c'] = pd.Categorical(df_comp['clinstat_2'], categories=[0, 1], ordered=True)

    cph = CoxPHFitter()
    model_params = {
        'duration_col': 'daysrz',
        'event_col': 'event_occurs',
        'formula': 'comparison_cat + clinstat_2_c'
    }
    if weights_col:
        model_params['weights_col'] = weights_col

    cph.fit(df_comp[['daysrz', 'event_occurs', 'comparison_cat', 'clinstat_2_c'] +
                    ([weights_col] if weights_col else [])], **model_params)

    stats = _extract_row_from_model(cph, 'comparison_cat')
    stats['Comparison'] = 'Overweight vs. well normal weight'
    stats['Dataset'] = dataset_name
    stats['Number of Observations'] = len(df_comp)
    results.append(stats)

    # 2. Obese vs. well normal weight, adjusted for illness severity
    df_comp = df[
        (df['bmi_cat'] == 4) |  # Obese
        ((df['bmi_cat'] == 2) & (df['clinstat_2'] == 0))  # Well normal weight
    ].copy()
    df_comp['comparison'] = df_comp.apply(
        lambda row: 1 if row['bmi_cat'] == 4 else 0, axis=1
    )
    df_comp['comparison_cat'] = pd.Categorical(df_comp['comparison'], categories=[0, 1], ordered=True)
    df_comp['clinstat_2_c'] = pd.Categorical(df_comp['clinstat_2'], categories=[0, 1], ordered=True)

    cph = CoxPHFitter()
    model_params['formula'] = 'comparison_cat + clinstat_2_c'

    cph.fit(df_comp[['daysrz', 'event_occurs', 'comparison_cat', 'clinstat_2_c'] +
                    ([weights_col] if weights_col else [])], **model_params)

    stats = _extract_row_from_model(cph, 'comparison_cat')
    stats['Comparison'] = 'Obese vs. well normal weight'
    stats['Dataset'] = dataset_name
    stats['Number of Observations'] = len(df_comp)
    results.append(stats)

    # 3. Unwell vs. well normal weight, adjusted for BMI
    df_comp = df[
        (df['clinstat_2'] == 1) |  # Unwell
        ((df['bmi_cat'] == 2) & (df['clinstat_2'] == 0))  # Well normal weight
    ].copy()
    df_comp['comparison'] = df_comp.apply(
        lambda row: 1 if row['clinstat_2'] == 1 else 0, axis=1
    )
    df_comp['comparison_cat'] = pd.Categorical(df_comp['comparison'], categories=[0, 1], ordered=True)

    # Create BMI dummy variables for adjustment
    df_comp['is_overweight'] = (df_comp['bmi_cat'] == 3).astype(int)
    df_comp['is_obese'] = (df_comp['bmi_cat'] == 4).astype(int)
    df_comp['is_overweight_cat'] = pd.Categorical(df_comp['is_overweight'], categories=[0, 1], ordered=True)
    df_comp['is_obese_cat'] = pd.Categorical(df_comp['is_obese'], categories=[0, 1], ordered=True)

    cph = CoxPHFitter()
    model_params['formula'] = 'comparison_cat + is_overweight_cat + is_obese_cat'

    cph.fit(df_comp[['daysrz', 'event_occurs', 'comparison_cat', 'is_overweight_cat', 'is_obese_cat'] +
                    ([weights_col] if weights_col else [])], **model_params)

    stats = _extract_row_from_model(cph, 'comparison_cat')
    stats['Comparison'] = 'Unwell vs. well normal weight'
    stats['Dataset'] = dataset_name
    stats['Number of Observations'] = len(df_comp)
    results.append(stats)

    # 4. Unwell overweight vs. well normal weight
    # Note: Cannot use interaction term due to perfect collinearity
    # Use simple univariate comparison instead
    df_comp = df[
        ((df['bmi_cat'] == 3) & (df['clinstat_2'] == 1)) |  # Unwell overweight
        ((df['bmi_cat'] == 2) & (df['clinstat_2'] == 0))  # Well normal weight
    ].copy()
    df_comp['comparison'] = df_comp.apply(
        lambda row: 1 if (row['bmi_cat'] == 3 and row['clinstat_2'] == 1) else 0, axis=1
    )
    df_comp['comparison_cat'] = pd.Categorical(df_comp['comparison'], categories=[0, 1], ordered=True)

    cph = CoxPHFitter()
    model_params['formula'] = 'comparison_cat'

    cph.fit(df_comp[['daysrz', 'event_occurs', 'comparison_cat'] +
                    ([weights_col] if weights_col else [])], **model_params)

    stats = _extract_row_from_model(cph, 'comparison_cat')
    stats['Comparison'] = 'Unwell overweight vs. well normal weight'
    stats['Dataset'] = dataset_name
    stats['Number of Observations'] = len(df_comp)
    results.append(stats)

    # 5. Unwell obese vs. well normal weight
    # Note: Cannot use interaction term due to perfect collinearity
    # Use simple univariate comparison instead
    df_comp = df[
        ((df['bmi_cat'] == 4) & (df['clinstat_2'] == 1)) |  # Unwell obese
        ((df['bmi_cat'] == 2) & (df['clinstat_2'] == 0))  # Well normal weight
    ].copy()
    df_comp['comparison'] = df_comp.apply(
        lambda row: 1 if (row['bmi_cat'] == 4 and row['clinstat_2'] == 1) else 0, axis=1
    )
    df_comp['comparison_cat'] = pd.Categorical(df_comp['comparison'], categories=[0, 1], ordered=True)

    cph = CoxPHFitter()
    model_params['formula'] = 'comparison_cat'

    cph.fit(df_comp[['daysrz', 'event_occurs', 'comparison_cat'] +
                    ([weights_col] if weights_col else [])], **model_params)

    stats = _extract_row_from_model(cph, 'comparison_cat')
    stats['Comparison'] = 'Unwell obese vs. well normal weight'
    stats['Dataset'] = dataset_name
    stats['Number of Observations'] = len(df_comp)
    results.append(stats)

    return results


def create_multivariate_bmi_illness_table(df_stepbd, df_catie, output_dir=None):
    """
    Create multivariate Cox regression table for BMI + Illness Severity.

    Same comparisons as Table 3, but with multivariate adjustment:
    - 5 rows per dataset (10 total)

    Returns:
    --------
    DataFrame : Results table
    """
    results = []

    # StepBD
    stepbd_stats = multivariate_bmi_illness(df_stepbd, 'StepBD')
    results.extend(stepbd_stats)

    # CATIE
    catie_stats = multivariate_bmi_illness(df_catie, 'CATIE')
    results.extend(catie_stats)

    # Create table
    results_df = pd.DataFrame(results)

    # FDR correction across all 10 rows
    _, p_adj = fdrcorrection([r['Uncorrected p'] for r in results], alpha=0.05)
    results_df['FDR-corrected p'] = p_adj

    # Reorder columns
    results_df = results_df[[
        'Dataset', 'Comparison', 'Number of Observations',
        'HR', 'HR, lower 95%', 'HR, upper 95%', 'z', 'Uncorrected p', 'FDR-corrected p'
    ]]

    # Sort by dataset and comparison order
    comparison_order = {
        'Overweight vs. well normal weight': 0,
        'Obese vs. well normal weight': 1,
        'Unwell vs. well normal weight': 2,
        'Unwell overweight vs. well normal weight': 3,
        'Unwell obese vs. well normal weight': 4
    }
    results_df['order'] = results_df['Dataset'].map({'StepBD': 0, 'CATIE': 1}) + \
                          results_df['Comparison'].map(comparison_order) / 10.0
    results_df = results_df.sort_values('order').drop(columns=['order']).reset_index(drop=True)

    if output_dir:
        results_df.to_csv(os.path.join(output_dir,
                         'Supplement_Table_6_Multivariate_BMI_Illness.csv'),
                         index=False)
        save_table_to_excel(results_df,
                          os.path.join(output_dir,
                                     'Supplement_Table_6_Multivariate_BMI_Illness.xlsx'),
                          sheet_name='Multivariate')

    logger.info("Multivariate BMI + Illness Severity table created")
    return results_df


# =============================================================================
# Main execution function
# =============================================================================

def run_supplementary_analyses(stepbd_smoking_path, catie_smoking_path,
                               stepbd_bmi_path, catie_bmi_path,
                               output_dir='../data4survivals_1/supplement_results'):
    """
    Run all supplementary analyses.

    Parameters:
    -----------
    stepbd_smoking_path, catie_smoking_path : str
        Paths to smoking datasets
    stepbd_bmi_path, catie_bmi_path : str
        Paths to BMI datasets
    output_dir : str
        Directory to save results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Loading datasets...")
    df_stepbd_smoking = pd.read_csv(stepbd_smoking_path)
    df_catie_smoking = pd.read_csv(catie_smoking_path)
    df_stepbd_bmi = pd.read_csv(stepbd_bmi_path)
    df_catie_bmi = pd.read_csv(catie_bmi_path)

    # Prepare data
    logger.info("Preparing datasets...")
    df_stepbd_smoking = prepare_data_for_visualization(df_stepbd_smoking)
    df_catie_smoking = prepare_data_for_visualization(df_catie_smoking)
    df_stepbd_bmi = prepare_data_for_visualization(df_stepbd_bmi)
    df_catie_bmi = prepare_data_for_visualization(df_catie_bmi)

    # 1. Population overlap
    logger.info("Creating population overlap report...")
    overlap_summary = create_population_overlap_report(
        df_stepbd_smoking, df_catie_smoking,
        df_stepbd_bmi, df_catie_bmi,
        output_dir
    )
    print("\nPopulation Overlap:")
    print(overlap_summary)

    # 2. KM curves for exclusive populations
    logger.info("Plotting KM curves for exclusive populations...")
    exclusive_medians = plot_km_exclusive_populations(
        df_stepbd_smoking, df_catie_smoking,
        df_stepbd_bmi, df_catie_bmi,
        output_dir
    )
    print("\nExclusive Population Median Survival Times:")
    print(exclusive_medians)

    # 3. Smoking main effect
    logger.info("Analyzing smoking main effect...")
    smoking_main = plot_smoking_main_effect(
        df_stepbd_smoking, df_catie_smoking, output_dir
    )
    print("\nSmoking Main Effect:")
    print(smoking_main)

    # 4. BMI main effect
    logger.info("Analyzing BMI main effect...")
    bmi_main = plot_bmi_main_effect(
        df_stepbd_bmi, df_catie_bmi, output_dir
    )
    print("\nBMI Main Effect:")
    print(bmi_main)

    # 5. Illness severity main effect
    logger.info("Analyzing illness severity main effect (using smoking data)...")
    illness_main = plot_illness_severity_main_effect(
        df_stepbd_smoking, df_catie_smoking, output_dir
    )
    print("\nIllness Severity Main Effect:")
    print(illness_main)

    # 6. Multivariate: Smoking + Illness
    logger.info("Creating multivariate table: Smoking + Illness Severity...")
    multivar_smoking = create_multivariate_smoking_illness_table(
        df_stepbd_smoking, df_catie_smoking, output_dir
    )
    print("\nMultivariate (Smoking + Illness):")
    print(multivar_smoking)

    # 7. Multivariate: BMI + Illness
    logger.info("Creating multivariate table: BMI + Illness Severity...")
    multivar_bmi = create_multivariate_bmi_illness_table(
        df_stepbd_bmi, df_catie_bmi, output_dir
    )
    print("\nMultivariate (BMI + Illness):")
    print(multivar_bmi)

    logger.info(f"All supplementary analyses completed. Results saved to {output_dir}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # Define paths
    stepbd_smoking_path = '../data4survivals_1/stepBD_smoking_final_event_occurrence.csv'
    catie_smoking_path = '../data4survivals_1/CATIE_smoking_final_event_occurrence.csv'
    stepbd_bmi_path = '../data4survivals_1/stepBD_bmi_final_event_occurrence.csv'
    catie_bmi_path = '../data4survivals_1/CATIE_bmi_final_event_occurrence.csv'

    # Run analyses
    run_supplementary_analyses(
        stepbd_smoking_path, catie_smoking_path,
        stepbd_bmi_path, catie_bmi_path
    )
