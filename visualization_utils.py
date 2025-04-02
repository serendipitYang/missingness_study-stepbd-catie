import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts


def prepare_data_for_visualization(df):
    """
    Prepare dataset for KM visualization by adding necessary categorical columns and weights.
    
    Parameters:
    df (DataFrame): Input DataFrame
    
    Returns:
    DataFrame: DataFrame with added categorical columns for analysis
    """
    # Add weight column for proper handling of repeated measures
    df = df.copy()
    df['weight'] = 1 / pow(df.groupby('subjectkey')['subjectkey'].transform('count'), 1)
    
    # Set up categorical variables for better visualization
    if 'smoker_yn_1' in df.columns:
        df['smoker_yn'] = df['smoker_yn_1'].apply(lambda x: 0 if x == 'non-smoker' else (2 if x == 'smoker' else 1))
        df['smoker_yn_c'] = pd.Categorical(df['smoker_yn'], categories=[0, 1, 2], ordered=True)
        
    if 'clinstat_1' in df.columns:
        df['clinstat_2'] = df['clinstat_1'].apply(lambda x: 0 if x == 0 else 1)
        df['clinstat_2_c'] = pd.Categorical(df['clinstat_2'], categories=[0, 1], ordered=True)
        
    if 'bmi_status' in df.columns and 'bmi_cat' not in df.columns:
        df['bmi_cat'] = df['bmi_status'].apply(lambda x: 2 if x == 'bmi2' else 
                                               (3 if x == 'bmi3' else 
                                                (4 if x == 'bmi456' else 0)))
    
    if 'bmi_cat' in df.columns:
        df['bmi_cat_1'] = df['bmi_cat'].apply(lambda x: 4 if x >= 4 else x)
        df['bmi_cat_c'] = pd.Categorical(df['bmi_cat_1'], categories=[2, 3, 4], ordered=True)
    
    return df


def plot_km_smoking_curves(df_stepBD, df_catie, highlight_times=None, filename=None):
    """
    Plot Kaplan-Meier curves for smokers vs non-smokers.
    
    Parameters:
    df_stepBD (DataFrame): StepBD dataset
    df_catie (DataFrame): CATIE dataset
    highlight_times (list, optional): Times to highlight on the curve
    filename (str, optional): Filename to save the plot
    """
    if highlight_times is None:
        highlight_times = [180, 360, 540, 720, 900, 1080]
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()
    
    for df_t, ax in zip([df_stepBD, df_catie], axes):
        df = df_t.copy()
        ind = np.where(axes == ax)[0][0]
        df = df[df['smoker_yn'] != 1]  # Remove changers
        
        # Create Kaplan-Meier objects for non-smokers and smokers
        kmf1 = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()
        
        kmf1.fit(df[df['smoker_yn_c'] == 0]['daysrz'], 
                event_observed=df[df['smoker_yn_c'] == 0]['event_occurs'], 
                weights=df[df['smoker_yn_c'] == 0]['weight'], 
                label='Non-Smokers')
        
        kmf2.fit(df[df['smoker_yn_c'] == 2]['daysrz'], 
                event_observed=df[df['smoker_yn_c'] == 2]['event_occurs'], 
                weights=df[df['smoker_yn_c'] == 2]['weight'], 
                label='Smokers')
        
        # Plot curves
        kmf1.plot(ax=ax, ci_alpha=0)
        kmf2.plot(ax=ax, ci_alpha=0)
        
        # Add highlight points
        for t in highlight_times:
            survival_prob_at_t1 = kmf1.predict(t)
            survival_prob_at_t2 = kmf2.predict(t)
            
            ax.scatter([t], [survival_prob_at_t1], marker='o', color='red')
            ax.annotate(f"P({t}d) = {survival_prob_at_t1:.2%}", 
                       (t, survival_prob_at_t1), 
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='left')
            
            ax.scatter([t], [survival_prob_at_t2], marker='x', color='green')
            ax.annotate(f"P({t}d) = {survival_prob_at_t2:.2%}", 
                       (t, survival_prob_at_t2), 
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='right')
        
        # Set titles and labels
        if ind == 0:
            ax.set_title('StepBD')
        if ind == 1:
            ax.set_title('CATIE')
            ax.set_xlim(0, 550)
            
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    plt.show()
    plt.close()


def plot_km_smoking_clinstat_curves(df_stepBD, df_catie, highlight_times=None, filename=None):
    """
    Plot Kaplan-Meier curves for smokers vs non-smokers stratified by clinical status.
    
    Parameters:
    df_stepBD (DataFrame): StepBD dataset
    df_catie (DataFrame): CATIE dataset
    highlight_times (list, optional): Times to highlight on the curve
    filename (str, optional): Filename to save the plot
    """
    if highlight_times is None:
        highlight_times = [180, 360, 540, 720, 900, 1080]
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()
    
    for df_t, ax in zip([df_stepBD, df_catie], axes):
        df = df_t.copy()
        df['clinstat_2_c'] = pd.Categorical(df['clinstat_2'], categories=[0, 1], ordered=True)
        ind = np.where(axes==ax)[0][0]
        
        # Create Kaplan-Meier objects for each group
        kmf_s0 = KaplanMeierFitter()
        kmf_s1 = KaplanMeierFitter()
        kmf_ns0 = KaplanMeierFitter()
        kmf_ns1 = KaplanMeierFitter()
        
        # Fit for well non-smokers
        kmf_s0.fit(df[(df['smoker_yn'] == 0) & (df['clinstat_2_c'] == 0)]['daysrz'], 
                  event_observed=df[(df['smoker_yn'] == 0) & (df['clinstat_2_c'] == 0)]['event_occurs'],
                  weights=df[(df['smoker_yn'] == 0) & (df['clinstat_2_c'] == 0)]['weight'], 
                  label='Well non-smokers')
        
        # Fit for unwell non-smokers
        kmf_s1.fit(df[(df['smoker_yn'] == 0) & (df['clinstat_2_c'] == 1)]['daysrz'], 
                  event_observed=df[(df['smoker_yn'] == 0) & (df['clinstat_2_c'] == 1)]['event_occurs'],
                  weights=df[(df['smoker_yn'] == 0) & (df['clinstat_2_c'] == 1)]['weight'],
                  label='Unwell non-smokers')
        
        # Fit for well smokers
        kmf_ns0.fit(df[(df['smoker_yn'] == 2) & (df['clinstat_2_c'] == 0)]['daysrz'], 
                   event_observed=df[(df['smoker_yn'] == 2) & (df['clinstat_2_c'] == 0)]['event_occurs'],
                   weights=df[(df['smoker_yn'] == 2) & (df['clinstat_2_c'] == 0)]['weight'],
                   label='Well smokers')
        
        # Fit for unwell smokers
        kmf_ns1.fit(df[(df['smoker_yn'] == 2) & (df['clinstat_2_c'] == 1)]['daysrz'], 
                   event_observed=df[(df['smoker_yn'] == 2) & (df['clinstat_2_c'] == 1)]['event_occurs'],
                   weights=df[(df['smoker_yn'] == 2) & (df['clinstat_2_c'] == 1)]['weight'],
                   label='Unwell smokers')
        
        # Plot curves
        kmf_s0.plot(ax=ax, ci_alpha=0)
        kmf_s1.plot(ax=ax, ci_alpha=0)
        kmf_ns0.plot(ax=ax, ci_alpha=0)
        kmf_ns1.plot(ax=ax, ci_alpha=0)
        
        if ind == 1:
            ax.legend()
        
        # Set titles and labels
        if ind == 0:
            ax.set_title('StepBD')
        if ind == 1:
            ax.set_title('CATIE')
            ax.set_xlim(0, 550)
            ax.set_ylim(0.4, 1)
            
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    plt.show()
    plt.close()


def plot_km_bmi_curves(df_stepBD, df_catie, highlight_times=None, filename=None):
    """
    Plot Kaplan-Meier curves for different BMI categories.
    
    Parameters:
    df_stepBD (DataFrame): StepBD dataset
    df_catie (DataFrame): CATIE dataset
    highlight_times (list, optional): Times to highlight on the curve
    filename (str, optional): Filename to save the plot
    """
    if highlight_times is None:
        highlight_times = [180, 360, 540, 720, 900, 1080]
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()
    
    for df_t, ax in zip([df_stepBD, df_catie], axes):
        df = df_t.copy()
        ind = np.where(axes==ax)[0][0]
        
        # Create Kaplan-Meier objects for different bmi_cat
        kmf1 = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()
        kmf3 = KaplanMeierFitter()
        
        # Fit for normal weight (18.5 <= BMI < 25)
        kmf1.fit(df[df['bmi_cat'] == 2]['daysrz'], 
                event_observed=df[df['bmi_cat'] == 2]['event_occurs'], 
                weights=df[df['bmi_cat'] == 2]['weight'], 
                label='18<=BMI<25')
        
        # Fit for overweight (25 <= BMI < 30)
        kmf2.fit(df[df['bmi_cat'] == 3]['daysrz'], 
                event_observed=df[df['bmi_cat'] == 3]['event_occurs'], 
                weights=df[df['bmi_cat'] == 3]['weight'], 
                label='25<=BMI<30')
        
        # Fit for obesity (BMI >= 30)
        kmf3.fit(df[df['bmi_cat'] >= 4]['daysrz'], 
                event_observed=df[df['bmi_cat'] >= 4]['event_occurs'], 
                weights=df[df['bmi_cat'] >= 4]['weight'], 
                label='BMI>=30')
        
        # Plot curves
        kmf1.plot(ax=ax, ci_alpha=0)
        kmf2.plot(ax=ax, ci_alpha=0)
        kmf3.plot(ax=ax, ci_alpha=0)
        
        if ind == 1:
            ax.legend()
        
        # Add highlight points
        for t in highlight_times:
            survival_prob_at_t1 = kmf1.predict(t)
            survival_prob_at_t2 = kmf2.predict(t)
            
            ax.scatter([t], [survival_prob_at_t1], marker='o', color='red')
            ax.annotate(f"P({t}d) = {survival_prob_at_t1:.2%}", 
                       (t, survival_prob_at_t1), 
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='left')
            
            ax.scatter([t], [survival_prob_at_t2], marker='x', color='green')
            ax.annotate(f"P({t}d) = {survival_prob_at_t2:.2%}", 
                       (t, survival_prob_at_t2), 
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='right')
        
        # Set titles and labels
        if ind == 0:
            ax.set_title('StepBD')
        if ind == 1:
            ax.set_title('CATIE')
            ax.set_xlim(0, 550)
            ax.set_ylim(0.45, 1)
            
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    plt.show()
    plt.close()


def plot_km_bmi_clinstat_curves(df_stepBD, df_catie, highlight_times=None, filename=None):
    """
    Plot Kaplan-Meier curves for different BMI categories stratified by clinical status.
    
    Parameters:
    df_stepBD (DataFrame): StepBD dataset
    df_catie (DataFrame): CATIE dataset
    highlight_times (list, optional): Times to highlight on the curve
    filename (str, optional): Filename to save the plot
    """
    if highlight_times is None:
        highlight_times = [180, 360, 540, 720, 900, 1080]
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()
    
    for df_t, ax in zip([df_stepBD, df_catie], axes):
        df = df_t.copy()
        df['clinstat_2_c'] = pd.Categorical(df['clinstat_2'], categories=[0, 1], ordered=True)
        ind = np.where(axes==ax)[0][0]
        
        # Create Kaplan-Meier objects
        kmf_sb2 = KaplanMeierFitter()
        kmf_sb3 = KaplanMeierFitter()
        kmf_sb456 = KaplanMeierFitter()
        kmf_nsb2 = KaplanMeierFitter()
        kmf_nsb3 = KaplanMeierFitter()
        kmf_nsb456 = KaplanMeierFitter()
        
        # Fit for well normal weight
        kmf_sb2.fit(df[(df['bmi_cat'] == 2) & (df['clinstat_2_c'] == 0)]['daysrz'], 
                   event_observed=df[(df['bmi_cat'] == 2) & (df['clinstat_2_c'] == 0)]['event_occurs'],
                   weights=df[(df['bmi_cat'] == 2) & (df['clinstat_2_c'] == 0)]['weight'],
                   label='Well 18<=BMI<25')
        
        # Fit for well overweight
        kmf_sb3.fit(df[(df['bmi_cat'] == 3) & (df['clinstat_2_c'] == 0)]['daysrz'], 
                   event_observed=df[(df['bmi_cat'] == 3) & (df['clinstat_2_c'] == 0)]['event_occurs'],
                   weights=df[(df['bmi_cat'] == 3) & (df['clinstat_2_c'] == 0)]['weight'],
                   label='Well 25<=BMI<30')
        
        # Fit for well obese
        kmf_sb456.fit(df[(df['bmi_cat'] >= 4) & (df['clinstat_2_c'] == 0)]['daysrz'], 
                     event_observed=df[(df['bmi_cat'] >= 4) & (df['clinstat_2_c'] == 0)]['event_occurs'],
                     weights=df[(df['bmi_cat'] >= 4) & (df['clinstat_2_c'] == 0)]['weight'],
                     label='Well BMI>=30')
        
        # Fit for unwell normal weight
        kmf_nsb2.fit(df[(df['bmi_cat'] == 2) & (df['clinstat_2_c'] == 1)]['daysrz'], 
                    event_observed=df[(df['bmi_cat'] == 2) & (df['clinstat_2_c'] == 1)]['event_occurs'],
                    weights=df[(df['bmi_cat'] == 2) & (df['clinstat_2_c'] == 1)]['weight'],
                    label='Unwell 18<=BMI<25')
        
        # Fit for unwell overweight
        kmf_nsb3.fit(df[(df['bmi_cat'] == 3) & (df['clinstat_2_c'] == 1)]['daysrz'], 
                    event_observed=df[(df['bmi_cat'] == 3) & (df['clinstat_2_c'] == 1)]['event_occurs'],
                    weights=df[(df['bmi_cat'] == 3) & (df['clinstat_2_c'] == 1)]['weight'],
                    label='Unwell 25<=BMI<30')
        
        # Fit for unwell obese
        kmf_nsb456.fit(df[(df['bmi_cat'] >= 4) & (df['clinstat_2_c'] == 1)]['daysrz'], 
                      event_observed=df[(df['bmi_cat'] >= 4) & (df['clinstat_2_c'] == 1)]['event_occurs'],
                      weights=df[(df['bmi_cat'] >= 4) & (df['clinstat_2_c'] == 1)]['weight'],
                      label='Unwell BMI>=30')
        
        # Plot curves
        kmf_sb2.plot(ax=ax, ci_alpha=0)
        kmf_nsb2.plot(ax=ax, ci_alpha=0)
        kmf_sb3.plot(ax=ax, ci_alpha=0)
        kmf_nsb3.plot(ax=ax, ci_alpha=0)
        kmf_sb456.plot(ax=ax, ci_alpha=0)
        kmf_nsb456.plot(ax=ax, ci_alpha=0)
        
        # Set titles and labels
        if ind == 0:
            ax.set_title('StepBD')
        if ind == 1:
            ax.set_title('CATIE')
            ax.set_xlim(0, 550)
            ax.set_ylim(0.3, 1)
            
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    plt.show()
    plt.close()
