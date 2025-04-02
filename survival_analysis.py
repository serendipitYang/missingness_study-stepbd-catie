import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from visualization_utils import (
    prepare_data_for_visualization,
    plot_km_smoking_curves,
    plot_km_smoking_clinstat_curves,
    plot_km_bmi_curves,
    plot_km_bmi_clinstat_curves
)
from cox_regression_utils import (
    run_basic_cox_model,
    run_smoking_pairwise_comparisons,
    run_bmi_pairwise_comparisons,
    adjust_pvalues,
    extract_cox_results
)


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
    
    # Limit CATIE data to 550 days for consistency with paper
    df_catie_smoking = df_catie_smoking[df_catie_smoking['daysrz'] <= 550]
    df_catie_bmi = df_catie_bmi[df_catie_bmi['daysrz'] <= 550]
    
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
    
    # Plot Kaplan-Meier curves
    plot_km_smoking_curves(
        df_stepBD_smoking, 
        df_catie_smoking, 
        filename=os.path.join(output_dir, 'smoking_km_curves.png') if output_dir else None
    )
    
    # Plot Kaplan-Meier curves stratified by clinical status
    plot_km_smoking_clinstat_curves(
        df_stepBD_smoking, 
        df_catie_smoking,
        filename=os.path.join(output_dir, 'smoking_clinstat_km_curves.png') if output_dir else None
    )
    
    # Run basic Cox models
    results['stepBD']['basic_cox'] = run_basic_cox_model(
        df_stepBD_smoking, 
        formula="smoker_yn_c + clinstat_2 + smoker_yn_c*clinstat_2",
        weights_col='weight'
    )
    
    results['CATIE']['basic_cox'] = run_basic_cox_model(
        df_catie_smoking, 
        formula="smoker_yn_c + clinstat_2 + smoker_yn_c*clinstat_2",
        weights_col='weight'
    )
    
    # Run pairwise comparisons
    results['stepBD']['pairwise'] = run_smoking_pairwise_comparisons(df_stepBD_smoking)
    results['CATIE']['pairwise'] = run_smoking_pairwise_comparisons(df_catie_smoking)
    
    # Extract p-values for multiple testing correction
    stepbd_pvals = [results['stepBD']['pairwise'][k]['p_value'] for k in results['stepBD']['pairwise']]
    catie_pvals = [results['CATIE']['pairwise'][k]['p_value'] for k in results['CATIE']['pairwise']]
    
    # All p-values combined for correction
    all_pvals = stepbd_pvals + catie_pvals
    
    # Adjust p-values
    _, adjusted_pvals = adjust_pvalues(all_pvals)
    
    # Store adjusted p-values
    results['adjusted_pvals'] = adjusted_pvals
    
    # Print summary of results
    print("StepBD Smoking Analysis:")
    for k, v in results['stepBD']['pairwise'].items():
        print(f"{k}: {extract_cox_results(v['model'], k)['formatted']}")
    
    print("\nCATIE Smoking Analysis:")
    for k, v in results['CATIE']['pairwise'].items():
        print(f"{k}: {extract_cox_results(v['model'], k)['formatted']}")
    
    return results


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
    
    # Plot Kaplan-Meier curves
    plot_km_bmi_curves(
        df_stepBD_bmi, 
        df_catie_bmi, 
        filename=os.path.join(output_dir, 'bmi_km_curves.png') if output_dir else None
    )
    
    # Plot Kaplan-Meier curves stratified by clinical status
    plot_km_bmi_clinstat_curves(
        df_stepBD_bmi, 
        df_catie_bmi,
        filename=os.path.join(output_dir, 'bmi_clinstat_km_curves.png') if output_dir else None
    )
    
    # Run basic Cox models
    results['stepBD']['basic_cox'] = run_basic_cox_model(
        df_stepBD_bmi, 
        formula="bmi_cat_c + clinstat_c + bmi_cat_c*clinstat_c",
        weights_col='weight'
    )
    
    results['CATIE']['basic_cox'] = run_basic_cox_model(
        df_catie_bmi, 
        formula="bmi_cat_c + clinstat_c + bmi_cat_c*clinstat_c",
        weights_col='weight'
    )
    
    # Run pairwise comparisons
    results['stepBD']['pairwise'] = run_bmi_pairwise_comparisons(df_stepBD_bmi)
    results['CATIE']['pairwise'] = run_bmi_pairwise_comparisons(df_catie_bmi)
    
    # Extract p-values for multiple testing correction
    stepbd_pvals = [results['stepBD']['pairwise'][k]['p_value'] for k in results['stepBD']['pairwise']]
    catie_pvals = [results['CATIE']['pairwise'][k]['p_value'] for k in results['CATIE']['pairwise']]
    
    # All p-values combined for correction
    all_pvals = stepbd_pvals + catie_pvals
    
    # Adjust p-values
    _, adjusted_pvals = adjust_pvalues(all_pvals)
    
    # Store adjusted p-values
    results['adjusted_pvals'] = adjusted_pvals
    
    # Print summary of results
    print("StepBD BMI Analysis:")
    for k, v in results['stepBD']['pairwise'].items():
        print(f"{k}: {extract_cox_results(v['model'], k)['formatted']}")
    
    print("\nCATIE BMI Analysis:")
    for k, v in results['CATIE']['pairwise'].items():
        print(f"{k}: {extract_cox_results(v['model'], k)['formatted']}")
    
    return results


def main():
    """
    Main function to run all analyses.
    """
    # File paths - replace with actual paths
    stepbd_smoking_path = "./stepBD_2missing_smoking_w_impute.csv"
    catie_smoking_path = "./CATIE_2missing_ptORcg_wellness_imputed_w_correction.csv"
    stepbd_bmi_path = "./StepBD_2missing_bmi_w_impute.csv"
    catie_bmi_path = "./CATIE_2missing_bmi_ptORcg_w_impute_bmi_corrected.csv"
    
    # Output directory
    output_dir = "./results"
    
    # Load and prepare datasets
    df_stepBD_smoking, df_catie_smoking, df_stepBD_bmi, df_catie_bmi = load_and_prepare_datasets(
        stepbd_smoking_path, catie_smoking_path, stepbd_bmi_path, catie_bmi_path
    )
    
    # Run smoking analysis
    smoking_results = run_smoking_analysis(df_stepBD_smoking, df_catie_smoking, output_dir)
    
    # Run BMI analysis
    bmi_results = run_bmi_analysis(df_stepBD_bmi, df_catie_bmi, output_dir)
    
    # Print adjusted p-values
    print("\nAdjusted p-values for smoking analysis:")
    for i, (comp_type, dataset) in enumerate([(ct, ds) for ds in ['stepBD', 'CATIE'] for ct in smoking_results[ds]['pairwise'].keys()]):
        orig_p = smoking_results[dataset]['pairwise'][comp_type]['p_value']
        adj_p = smoking_results['adjusted_pvals'][i]
        print(f"{dataset} {comp_type}: Original p={orig_p:.6f}, Adjusted p={adj_p:.6f}")
    
    print("\nAdjusted p-values for BMI analysis:")
    for i, (comp_type, dataset) in enumerate([(ct, ds) for ds in ['stepBD', 'CATIE'] for ct in bmi_results[ds]['pairwise'].keys()]):
        orig_p = bmi_results[dataset]['pairwise'][comp_type]['p_value']
        adj_p = bmi_results['adjusted_pvals'][i]
        print(f"{dataset} {comp_type}: Original p={orig_p:.6f}, Adjusted p={adj_p:.6f}")


if __name__ == "__main__":
    main()
