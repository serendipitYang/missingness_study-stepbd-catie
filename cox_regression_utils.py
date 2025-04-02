import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import fdrcorrection


def run_basic_cox_model(df, formula="clinstat_2", weights_col=None):
    """
    Run a basic Cox proportional hazards model.
    
    Parameters:
    df (DataFrame): Input DataFrame
    formula (str): Formula for the Cox model
    weights_col (str, optional): Column name for weights
    
    Returns:
    CoxPHFitter: Fitted Cox model
    """
    cph = CoxPHFitter()
    
    # Prepare model parameters
    model_params = {
        'duration_col': 'daysrz',
        'event_col': 'event_occurs',
        'formula': formula
    }
    
    # Add weights if provided
    if weights_col:
        model_params['weights_col'] = weights_col
    
    # Fit the model
    cph.fit(df, **model_params)
    
    return cph


def run_pairwise_cox_model(df, comparison_type, ref_group_params, comp_group_params, weights_col=None):
    """
    Run a Cox model comparing two specific groups.
    
    Parameters:
    df (DataFrame): Input DataFrame
    comparison_type (str): Name for the comparison
    ref_group_params (dict): Parameters defining the reference group
    comp_group_params (dict): Parameters defining the comparison group
    weights_col (str, optional): Column name for weights
    
    Returns:
    tuple: (CoxPHFitter, DataFrame with the comparison column)
    """
    df = df.copy()
    
    # Create the comparison column
    df[comparison_type] = df.apply(
        lambda row: 1 if all(row[k] == v for k, v in comp_group_params.items()) 
                  else (0 if all(row[k] == v for k, v in ref_group_params.items()) 
                      else np.nan), 
        axis=1
    )
    
    # Convert to categorical
    df[comparison_type] = pd.Categorical(df[comparison_type], categories=[0, 1], ordered=True)
    
    # Drop rows with NaN in the comparison column
    df_filtered = df.dropna(subset=[comparison_type])
    
    # Fit the Cox model
    cph = CoxPHFitter()
    
    # Prepare model parameters
    model_params = {
        'duration_col': 'daysrz',
        'event_col': 'event_occurs',
        'formula': comparison_type
    }
    
    # Add weights if provided
    if weights_col:
        model_params['weights_col'] = weights_col
    
    # Fit the model
    cph.fit(df_filtered[['daysrz', 'event_occurs', comparison_type] + 
                        ([weights_col] if weights_col else [])], **model_params)
    
    return cph, df_filtered


def run_smoking_pairwise_comparisons(df, with_weights=True):
    """
    Run all pairwise comparisons for smoking status and clinical status.
    
    Parameters:
    df (DataFrame): Input DataFrame
    with_weights (bool): Whether to use weights in the models
    
    Returns:
    dict: Dictionary of CoxPHFitter models and p-values
    """
    weights_col = 'weight' if with_weights else None
    results = {}
    
    # 1. Smoker vs. Well Non-smoker
    s_vs_nsw_cph, _ = run_pairwise_cox_model(
        df, 
        's_vs_nsw',
        {'smoker_yn_c': 0, 'clinstat_2': 0},  # Well non-smoker
        {'smoker_yn_c': 2},  # Smoker
        weights_col
    )
    results['s_vs_nsw'] = {
        'model': s_vs_nsw_cph,
        'p_value': s_vs_nsw_cph.summary.loc['s_vs_nsw[T.1]', 'p']
    }
    
    # 2. Unwell vs. Well Non-smoker
    uw_vs_nsw_cph, _ = run_pairwise_cox_model(
        df, 
        'uw_vs_nsw',
        {'smoker_yn_c': 0, 'clinstat_2': 0},  # Well non-smoker
        {'clinstat_2': 1},  # Unwell
        weights_col
    )
    results['uw_vs_nsw'] = {
        'model': uw_vs_nsw_cph,
        'p_value': uw_vs_nsw_cph.summary.loc['uw_vs_nsw[T.1]', 'p']
    }
    
    # 3. Unwell Smoker vs. Well Non-smoker
    suw_vs_nsw_cph, _ = run_pairwise_cox_model(
        df, 
        'suw_vs_nsw',
        {'smoker_yn_c': 0, 'clinstat_2': 0},  # Well non-smoker
        {'smoker_yn_c': 2, 'clinstat_2': 1},  # Unwell smoker
        weights_col
    )
    results['suw_vs_nsw'] = {
        'model': suw_vs_nsw_cph,
        'p_value': suw_vs_nsw_cph.summary.loc['suw_vs_nsw[T.1]', 'p']
    }
    
    return results


def run_bmi_pairwise_comparisons(df, with_weights=True):
    """
    Run all pairwise comparisons for BMI categories and clinical status.
    
    Parameters:
    df (DataFrame): Input DataFrame
    with_weights (bool): Whether to use weights in the models
    
    Returns:
    dict: Dictionary of CoxPHFitter models and p-values
    """
    weights_col = 'weight' if with_weights else None
    results = {}
    
    # 1. Overweight vs. Well Normal weight
    o_vs_nw_cph, _ = run_pairwise_cox_model(
        df, 
        'o_vs_nw',
        {'bmi_cat_1': 2, 'clinstat_2': 0},  # Well normal weight
        {'bmi_cat_1': 3},  # Overweight
        weights_col
    )
    results['o_vs_nw'] = {
        'model': o_vs_nw_cph,
        'p_value': o_vs_nw_cph.summary.loc['o_vs_nw[T.1]', 'p']
    }
    
    # 2. Obese vs. Well Normal weight
    obe_vs_nw_cph, _ = run_pairwise_cox_model(
        df, 
        'obe_vs_nw',
        {'bmi_cat_1': 2, 'clinstat_2': 0},  # Well normal weight
        {'bmi_cat_1': 4},  # Obese
        weights_col
    )
    results['obe_vs_nw'] = {
        'model': obe_vs_nw_cph,
        'p_value': obe_vs_nw_cph.summary.loc['obe_vs_nw[T.1]', 'p']
    }
    
    # 3. Unwell vs. Well Normal weight
    uw_vs_nw_cph, _ = run_pairwise_cox_model(
        df, 
        'uw_vs_nw',
        {'bmi_cat_1': 2, 'clinstat_2': 0},  # Well normal weight
        {'clinstat_2': 1},  # Unwell
        weights_col
    )
    results['uw_vs_nw'] = {
        'model': uw_vs_nw_cph,
        'p_value': uw_vs_nw_cph.summary.loc['uw_vs_nw[T.1]', 'p']
    }
    
    # 4. Unwell Overweight vs. Well Normal weight
    ouw_vs_nw_cph, _ = run_pairwise_cox_model(
        df, 
        'ouw_vs_nw',
        {'bmi_cat_1': 2, 'clinstat_2': 0},  # Well normal weight
        {'bmi_cat_1': 3, 'clinstat_2': 1},  # Unwell overweight
        weights_col
    )
    results['ouw_vs_nw'] = {
        'model': ouw_vs_nw_cph,
        'p_value': ouw_vs_nw_cph.summary.loc['ouw_vs_nw[T.1]', 'p']
    }
    
    # 5. Unwell Obese vs. Well Normal weight
    obuw_vs_nw_cph, _ = run_pairwise_cox_model(
        df, 
        'obuw_vs_nw',
        {'bmi_cat_1': 2, 'clinstat_2': 0},  # Well normal weight
        {'bmi_cat_1': 4, 'clinstat_2': 1},  # Unwell obese
        weights_col
    )
    results['obuw_vs_nw'] = {
        'model': obuw_vs_nw_cph,
        'p_value': obuw_vs_nw_cph.summary.loc['obuw_vs_nw[T.1]', 'p']
    }
    
    return results


def adjust_pvalues(pvalues, method='fdr_bh'):
    """
    Adjust p-values for multiple comparisons.
    
    Parameters:
    pvalues (list): List of p-values to adjust
    method (str): Method for adjustment ('fdr_bh' for Benjamini-Hochberg)
    
    Returns:
    tuple: (rejected hypotheses, adjusted p-values)
    """
    if method == 'fdr_bh':
        rejected, pvals_corrected = fdrcorrection(pvalues, alpha=0.05)
        return rejected, pvals_corrected
    else:
        raise ValueError(f"Adjustment method {method} not supported")


def extract_cox_results(model, var_name):
    """
    Extract key results from a Cox model for a specific variable.
    
    Parameters:
    model (CoxPHFitter): Fitted Cox model
    var_name (str): Variable name to extract results for
    
    Returns:
    dict: Dictionary with hazard ratio, confidence intervals, and p-value
    """
    summary = model.summary
    if var_name not in summary.index:
        var_name = f"{var_name}[T.1]"  # Try with categorical syntax
    
    if var_name not in summary.index:
        return None
        
    hr = summary.loc[var_name, 'exp(coef)']
    hr_lower = summary.loc[var_name, 'exp(coef) lower 95%']
    hr_upper = summary.loc[var_name, 'exp(coef) upper 95%']
    p_value = summary.loc[var_name, 'p']
    
    return {
        'hazard_ratio': hr,
        'ci_lower': hr_lower,
        'ci_upper': hr_upper,
        'p_value': p_value,
        'formatted': f"HR {hr:.2f} [{hr_lower:.2f}, {hr_upper:.2f}], p={p_value:.4f}"
    }


def two_proportion_ztest(success1, size1, success2, size2, alternative='two-sided'):
    """
    Perform a two-proportion z-test using statsmodels.

    Args:
        success1 (int): Number of successes in the first sample.
        size1 (int): Size of the first sample.
        success2 (int): Number of successes in the second sample.
        size2 (int): Size of the second sample.
        alternative (str): Type of alternative hypothesis ('two-sided', 'smaller', 'larger').

    Returns:
        tuple: z-statistic and p-value
    """
    import statsmodels.api as sm
    
    count = np.array([success1, success2])
    nobs = np.array([size1, size2])
    zstat, pval = sm.stats.proportions_ztest(count, nobs, alternative=alternative)
    
    return zstat, pval