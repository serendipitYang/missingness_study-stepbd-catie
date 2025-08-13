import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import logging
import warnings
from collections import Counter
from utils import (
    GetSMK,
    GetClinstat2,
    bmistatus2cat,
    prepare_data_for_visualization,
    plot_km_smoking_clinstat_curves,
    plot_km_bmi_clinstat_curves,
    run_pairwise_cox_model,
    run_smoking_pairwise_comparisons,
    run_bmi_pairwise_comparisons,
    make_table2,
    make_table3
)
# ------------------------------------------------------------------------------
# Configure logger and disable warnings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
# ------------------------------------------------------------------------------


def load_and_prepare_datasets(stepbd_smoking_path, catie_smoking_path, 
                             stepbd_bmi_path, catie_bmi_path):
    """
    Load and prepare datasets for analysis.
    
    Parameters:
    stepbd_smoking_path (str): Path to StepBD smoking dataset
    catie_smoking_path (str): Path to CATIE smoking dataset
    stepbd_bmi_path (str): Path to StepBD BMI dataset
    catie_bmi_path (str): Path to CATIE BMI dataset
    
    Returns:
    tuple: Tuple of prepared DataFrames
    """
    # Load smoking datasets
    df_stepBD_smoking = pd.read_csv(stepbd_smoking_path)
    df_catie_smoking = pd.read_csv(catie_smoking_path)
    
    # Load BMI datasets
    df_stepBD_bmi = pd.read_csv(stepbd_bmi_path)
    df_catie_bmi = pd.read_csv(catie_bmi_path)
    
    # Prepare datasets for visualization
    df_stepBD_smoking = prepare_data_for_visualization(df_stepBD_smoking)
    df_catie_smoking = prepare_data_for_visualization(df_catie_smoking)
    df_stepBD_bmi = prepare_data_for_visualization(df_stepBD_bmi)
    df_catie_bmi = prepare_data_for_visualization(df_catie_bmi)
    
    return (df_stepBD_smoking, df_catie_smoking, 
            df_stepBD_bmi, df_catie_bmi)


def run_smoking_analysis(df_stepBD_smoking, df_catie_smoking, output_dir=None):
    """
    Run comprehensive analysis on smoking datasets.
    
    Parameters:
    df_stepBD_smoking (DataFrame): StepBD smoking dataset
    df_catie_smoking (DataFrame): CATIE smoking dataset
    output_dir (str, optional): Directory to save output files
    
    Returns:
    tuple: Dictionary of results for both datasets
    """
    results = {
        'stepBD': {},
        'CATIE': {}
    }
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    # Plot Kaplan-Meier Smoking effect curves stratified by clinical status
    smoking_median_days = plot_km_smoking_clinstat_curves(
        df_stepBD_smoking, 
        df_catie_smoking,
        filename=os.path.join(output_dir, 'Figure_2_smoking_clinstat_km_curves.png') if output_dir else None
    )
    print("Median days to event (smoking analysis):")
    print(smoking_median_days)

    # Run pairwise comparisons for smoking status straitified by clinical status
    logger.info("Making Table 2, running pairwise comparisons for smoking status...")
    table2 = make_table2(
        df_stepBD_smoking, df_catie_smoking, 
        out_csv=os.path.join(output_dir, 'Table_2_smoking_pairwise_comparisons.csv'),
        out_excel=os.path.join(output_dir, 'Table_2_smoking_pairwise_comparisons.xlsx')
        )
    
    return table2


def run_bmi_analysis(df_stepBD_bmi, df_catie_bmi, output_dir=None):
    """
    Run comprehensive analysis on BMI datasets.
    
    Parameters:
    df_stepBD_bmi (DataFrame): StepBD BMI dataset
    df_catie_bmi (DataFrame): CATIE BMI dataset
    output_dir (str, optional): Directory to save output files
    
    Returns:
    tuple: Dictionary of results for both datasets
    """
    results = {
        'stepBD': {},
        'CATIE': {}
    }
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot Kaplan-Meier curves, BMI category stratified by clinical status
    bmi_median_days = plot_km_bmi_clinstat_curves(
        df_stepBD_bmi, 
        df_catie_bmi, 
        filename=os.path.join(output_dir, 'Figure_3_bmi_clinstat_km_curves.png') if output_dir else None
    )
    print("Median days to event (bmi analysis):")
    print(bmi_median_days)

    
    # Run pairwise comparisons for smoking status straitified by clinical status
    logger.info("Making Table 2, running pairwise comparisons for smoking status...")
    table3 = make_table3(
        df_stepBD_bmi, df_catie_bmi, 
        out_csv=os.path.join(output_dir, 'Table_3_BMI_pairwise_comparisons.csv'),
        out_excel=os.path.join(output_dir, 'Table_3_BMI_pairwise_comparisons.xlsx')
        )
    
    return table3


def main():
    """
    Main function to run all analyses.
    """
    # File paths - replace with actual paths
    # stepbd_smoking_path = '../data4survivals_1/stepBD_smoking_final_event_occurrence.csv'
    # catie_smoking_path = "../data4survivals_1/CATIE_smoking_final_event_occurrence.csv"
    # stepbd_bmi_path = '../data4survivals_1/stepBD_bmi_final_event_occurrence.csv'
    # catie_bmi_path = '../data4survivals_1/CATIE_bmi_final_event_occurrence.csv'
    stepbd_smoking_path = '../data4survivals/stepBD_2missing_w_impute_w_correction_relabeling_w_updated_smokers_baseline.csv'
    catie_smoking_path = '../data4survivals/CATIE_2missing_ptORcg_wellness_imputed_w_correction_w_updated_smokers_baseline.csv'
    stepbd_bmi_path = '../data4survivals/StepBD_2missing_bmi_w_impute_relabelled_baseline_imputeless.csv'
    catie_bmi_path = '../data4survivals/CATIE_2missing_bmi_ptORcg_w_impute_bmi_corrected_baseline.csv'
    
    
    # Output directory
    output_dir = '../data4survivals_1/results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    # Load and prepare datasets
    logger.info("Loading and preparing datasets...")
    df_stepBD_smoking, df_catie_smoking, df_stepBD_bmi, df_catie_bmi = load_and_prepare_datasets(
        stepbd_smoking_path, catie_smoking_path, stepbd_bmi_path, catie_bmi_path
    )
    
    # Run smoking analysis
    logger.info("Running smoking analysis...")
    smoking_results = run_smoking_analysis(df_stepBD_smoking, df_catie_smoking, output_dir)
    

    # Run BMI analysis
    logger.info("Running BMI analysis...")
    bmi_results = run_bmi_analysis(df_stepBD_bmi, df_catie_bmi, output_dir)
    
    logger.info("All analyses completed. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
