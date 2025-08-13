import pandas as pd
import numpy as np
from tqdm import tqdm
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from statsmodels.stats.multitest import fdrcorrection

def bmi2cate(bmi_value):
    if pd.isna(bmi_value):
        return np.nan
    if bmi_value<18.5:
        return "Underweight"
    if bmi_value<25:
        return "Normal weight"
    if bmi_value<30:
        return "Overweight"
    return "Obese"

def locf_imputation(df, columns):
    """
    Impute missing values using Last Observation Carried Forward (LOCF) method, a
    with additional handling for leading NaN values and same-day imputation.
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    columns (list): List of column names to impute.

    Returns:
    DataFrame: The DataFrame with imputed values.
    """
    # Sort the DataFrame by subjectkey and daysrz to ensure correct order
    df = df.sort_values(by=['subjectkey', 'daysrz']).copy()
    df = df.reset_index()
    del df['index']
    
    # Loop through each subject
    for subject in tqdm(df['subjectkey'].unique()):
        subject_data = df[df['subjectkey'] == subject]

        # Loop through each specified column
        for col in columns:
            # Find the first non-NaN value
            first_valid_index = subject_data[col].first_valid_index()
            if first_valid_index is not None:
                first_value = subject_data[col].loc[first_valid_index]
                
                # Fill all leading NaNs with the first valid value
                subject_data[col] = subject_data[col].fillna(method='ffill').fillna(first_value)

            # Iterate through rows to handle remaining NaNs
            for idx in subject_data.index:
                if pd.isna(subject_data.loc[idx, col]):
                    # Get the current day
                    current_day = subject_data.loc[idx, 'daysrz']
                    
                    # Find rows with the same day
                    same_day_data = subject_data[subject_data['daysrz'] == current_day]
                    
                    # Check if the column has a non-NaN value in the same-day rows
                    same_day_value = same_day_data[col].dropna()
                    if not same_day_value.empty:
                        # Impute the NaN with the same-day value
                        subject_data.loc[idx, col] = same_day_value.iloc[0]
            
            # Apply LOCF to fill any remaining NaNs
            subject_data[col] = subject_data[col].fillna(method='ffill')
        
        # Update the main DataFrame with the imputed subject data
        df.loc[subject_data.index, columns] = subject_data[columns]

    return df

def select_least_nan_row(group):
    # Count NaNs across all columns for each row in the group
    nan_counts = group.isnull().sum(axis=1)
    # Find the index of the row with the minimum number of NaNs
    min_nan_index = nan_counts.idxmin()
    return group.loc[[min_nan_index]]

def map_clinstat(clinstat):
    if clinstat in [1, 2, 3, 4]:
        return 2
    elif clinstat in [5, 6]:
        return 1
    elif clinstat in [7, 8]:
        return 0
    else:
        return None  # or a default value if needed
    
def get_event(Alist):
    temp_ind = 0
    y = [0 for _ in range(len(Alist))]
    for idx, l in enumerate(Alist):
        if temp_ind==2: # How many missing visits we need to identify a dropout
            y[idx]=1
        else:
            if l==0:
                temp_ind+=1
    if 1 not in y:
        y[len(y)-1]=1
    return y

def get_visit_id(visit):
    y = [0 for _ in range(len(visit))]
    for idx,v in enumerate(visit):
        if pd.isna(v):
            y[idx] = y[idx-1]
            continue
        if ' month' in v:
            y[idx] = int(v.split(" month")[0])
        elif idx>0:
            y[idx] = y[idx-1]
    return y

def get_event_by_visit(Alist):
    temp_ind = 0
    y = [0 for _ in range(len(Alist))]
    for idx, l in enumerate(Alist):
        if temp_ind==2: # How many missing visits are needed to identify a dropout, 2 in this case
            y[idx]=1
        elif idx>0:
            if l-Alist[idx-1]>3:
                temp_ind+=1
    if 1 not in y:
        y[len(y)-1]=1
    return y

def compute_schedule_visit(df, time_window = 15):
    # Define the visit schedule in terms of days
    visit_schedule = {
        "baseline": 0,
        "3 month": 90,
        "6 month": 180,
        "9 month": 270,
        "12 month": 360,
        "18 month": 540,
        "24 month": 720,
        "30 month": 900,
        "36 month": 1080,
        "42 month": 1260,
        "48 month": 1440,
        "54 month": 1620,
        "60 month": 1800
    }

    # Initialize a new column for schedule_visit with None
    df['schedule_visit'] = 'missed'
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Only process rows where is_schedule == 1
        if row['is_scheduled'] == 1:
            # Check if daysrz falls within the window of any scheduled visit
            for visit, day in visit_schedule.items():
                if abs(row['daysrz'] - day) <= time_window:
                    df.at[index, 'schedule_visit'] = visit
                    break  # Assign only the first matching visit
    
    return df

panss_cols = ['collection_id', 'panss01_id', 'dataset_id', 'subjectkey', 'src_subject_id', 'interview_age', 'interview_date', 'sex', 'panss_1', 'panss_2', 'panss_3', 'panss_4', 'panss_5', 'panss_6', 'panss_7', 'panss_8', 'panss_9', 'panss_10', 'panss_11', 'panss_12', 'panss_13', 'panss_14', 'panss_15', 'panss_16', 'panss_17', 'panss_18', 'panss_19', 'panss_20', 'panss_21', 'panss_22', 'panss_23', 'panss_24', 'panss_25', 'panss_26', 'panss_27', 'panss_28', 'panss_29', 'panss_30', 'panss_31', 'panss_32', 'panss_33', 'panss_34', 'panss_35', 'panss_36', 'panss_37', 'panss_38', 'panss_39', 'panss_40', 'panss_41', 'panss_42', 'panss_43', 'panss_44', 'panss_45', 'panss_46', 'panss_47', 'panss_48', 'panss_49', 'panss_50', 'panss_51', 'panss_52', 'panss_53', 'panss_54', 'panss_55', 'panss_56', 'panss_57', 'panss_58', 'panss_59', 'panss_60', 'panss_61', 'panss_62', 'panss_63', 'panss_64', 'panss_65', 'panss_66', 'panss_67', 'panss_68', 'panss_69', 'panss_70', 'panss_71', 'panss_72', 'panss_73', 'panss_74', 'panss_75', 'panss_76', 'panss_77', 'panss_78', 'panss_79', 'panss_80', 'panss_81', 'panss_82', 'panss_83', 'panss_84', 'panss_85', 'panss_86', 'panss_87', 'panss_88', 'panss_89', 'panss_90', 'panss_91', 'panss_92', 'panss_93', 'panss_94', 'panss_95', 'panss_96', 'panss_97', 'panss_98', 'panss_99', 'panss_100', 'panss_101', 'panss_102', 'panss_103', 'panss_104', 'panss_105', 'panss_106', 'panss_107', 'panss_108', 'panss_109', 'panss_110', 'panss_111', 'panss_112', 'panss_113', 'panss_114', 'panss_115', 'panss_116', 'panss_117', 'panss_118', 'panss_119', 'panss_120', 'panss_121', 'panss_122', 'panss_123', 'panss_124', 'panss_125', 'panss_126', 'panss_127', 'panss_128', 'panss_129', 'panss_130', 'panss_131', 'panss_132', 'panss_132a', 'panss_133', 'panss_134', 'panss_135', 'panss_136', 'panss_137', 'panss_138', 'panss_139', 'panss_140', 'panss_141', 'panss_142', 'panss_143', 'panss_144', 'panss_145', 'panss_146', 'panss_147', 'panss_148', 'panss_149', 'panss_150', 'panss_151', 'panss_152', 'panss_153', 'panss_154', 'panss_155', 'panss_156', 'panss_157', 'panss_158', 'panss_159', 'panss_160', 'panss_161', 'panss_162', 'panss_163', 'panss_164', 'panss_165', 'panss_166', 'panss_167', 'panss_168', 'panss_169', 'panss_170', 'panss_171', 'panss_172', 'panss_173', 'panss_174', 'panss_175', 'panss_176', 'panss_177', 'panss_178', 'panss_179', 'panss_180', 'panss_181', 'panss_182', 'panss_183', 'panss_184', 'panss_185', 'panss_186', 'panss_187', 'panss_188', 'panss_189', 'panss_190', 'panss_191', 'panss_192', 'panss_193', 'panss_194', 'panss_195', 'panss_196', 'panss_197', 'panss_198', 'panss_199', 'panss_200', 'panss_201', 'panss_202', 'panss_203', 'panss_204', 'panss_205', 'panss_206', 'panss_207', 'panss_208', 'panss_209', 'panss_210', 'panss_211', 'panss_212', 'panss_213', 'panss_214', 'panss_215', 'panss_216', 'panss_217', 'panss_218', 'panss_219', 'panss_220', 'panss_five_cognitive', 'panss_five_emotion', 'panss_five_hostility', 'panss_five_negative', 'panss_five_positive', 'panss_general', 'panss_negative', 'panss_positive', 'panss_total', 'pos_p1', 'pos_p2', 'pos_p3', 'pos_p4', 'pos_p5', 'pos_p6', 'pos_p7', 'neg_n1', 'neg_n2', 'neg_n3', 'neg_n4', 'neg_n5', 'neg_n6', 'neg_n7', 'gps_g1', 'gps_g2', 'gps_g3', 'gps_g4', 'gps_g5', 'gps_g6', 'gps_g7', 'gps_g8', 'gps_g9', 'gps_g10', 'gps_g11', 'gps_g12', 'gps_g13', 'gps_g14', 'gps_g15', 'gps_g16', 'phase_ct', 'base4', 'b1_ppos', 'b1_pneg', 'b1_ppsy', 'cut_pans', 'cut_pos', 'cut_neg', 'cut_gen', 'c1_panss', 'c1_ppos', 'c1_pneg', 'c1_ppsy', 'visitid', 'visit', 'truncvis', 'visday', 'base1', 'base1b', 'base2', 'base3', 'last1', 'b1_panss', 'study_id', 'study_condition', 'time_point', 'days_baseline', 'aescode', 'bafinfo', 'comments_misc', 'panss_3fgen', 'panss_3fneg', 'panss_3fpos', 'fseqno', 'version_form', 'daysrz', 'ml30pnsp', 'ml30pnsn', 'ml30pnsg', 'chpanss', 'panss_c', 'ml30pns', 'site', 'm3distsx', 'm5deptsx', 'm4exctsx', 'm1negtsx', 'm2postsx', 'sxrem', 'pantsx', 'monthsbl', 'w3distsx', 'w4exctsx', 'w5deptsx', 'w2negtsx', 'w1postsx', 'panpstsx', 'panngtsx', 'w1posts', 'w2negts', 'w3dists', 'w4excts', 'w5depts', 'm1negts', 'm2posts', 'm3dists', 'm4excts', 'm5depts', 'panegtsx', 'study_level', 'ttt_arms', 'week', 'rand_num', 'pan_anger', 'pan_delaygrat', 'pan_afflabil', 'panss_supplementary', 'dode', 't1wk', 'maj_change', 'medchange', 'hospital_month', 'panss_inaff', 'panss_3_comp', 'panss_5_dis', 'panss_cluster_ang', 'panss_cluster_tht', 'panss_cluster_act', 'panss_cluster_par', 'panss_cluster_dep', 'study', 'assessment_complete', 'pan_disorient_2', 'pan_ward', 'pan_hospital_staff', 'pan_president', 'pan_dis_hos', 'pan_dis_pro', 'vr_comfort_move_date', 'panss_experimental', 'panss_expressive', 'panss2_dis', 'collection_title']
clgry_cols = ['collection_id', 'clgry01_id', 'dataset_id', 'subjectkey', 'src_subject_id', 'interview_age', 'interview_date', 'sex', 'visitid', 'visit', 'truncvis', 'phase_ct', 'visday', 'calg1', 'calg2', 'calg3', 'calg4', 'calg5', 'calg6', 'calg7', 'calg8', 'calg9', 'calg_ts', 'base1', 'last1', 'b1_calg', 'c1_calg', 'site', 'visit_name', 'subjecttype', 'elig_inclusion_check', 'dataquality', 'calg_s1', 'calg10', 'calg_s2', 'suicidality1', 'suicidality2', 'cdsstsx', 'days_baseline', 'monthsbl', 't1wk', 'assessment_complete', 'comments_misc', 'collection_title']
cig_cols = ['collection_id', 'cgis01_id', 'dataset_id', 'subjectkey', 'src_subject_id', 'interview_age', 'interview_date', 'sex', 'visitid', 'visit', 'truncvis', 'phase_ct', 'visday', 'cs01', 'cs02a', 'cs02aa', 'cs02b', 'cs02ba', 'cs02c', 'cs02ca', 'cs02d', 'cs02da', 'cs02e', 'cs02ea', 'cs02f', 'cs02fa', 'cs03', 'cs04', 'cs05', 'cs06', 'cs07', 'cs08', 'cs09', 'cs10', 'cs11', 'cs12', 'cs13', 'cscigs', 'cs14', 'cs15', 'cs16', 'base1', 'last1', 'b1_cgis', 'b1_resp', 'c1_cgis', 'cgisresp', 'site', 'dsmvers', 'date_updated', 'session_id', 'completed', 'aefther', 'cgas', 'bsit0', 'days_baseline', 'well_04', 'imputation', 'comments', 'monthsbl', 'version_form', 'cgither', 'week', 'cgi_s_dep', 'cgi_s_manic', 'cgi_s_overall', 'collection_title']

def map_clinstat_panss_or_calg(df, is_panss=True):
    """
    Map clinical status based on PANSS total or Calgary score. Used in CATIE data processing.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    is_panss (bool): Flag to determine if the mapping is for PANSS or Calgary score
    
    Returns:
    pd.Series: Mapped clinical status
    """
    def map_panss_clinstat(pansstotal):
        if np.isnan(pansstotal):
            return np.nan
        if pansstotal < 58:
            return 0
        else:
            return 1

    def map_calg_clinstat(score):
        if np.isnan(score):
            return np.nan
        if score <= 6:
            return 0
        else:
            return 1

    def panss_OR_calg2clinstat(df):
        if np.isnan(df['clinstat_pt']) and np.isnan(df['clinstat_cg']):
            return np.nan
        if 1 in (df['clinstat_pt'], df['clinstat_cg']):
            return 1
        else:
            return 0

    if is_panss:
        return df['panss_total'].apply(map_panss_clinstat)
    elif is_panss is False:
        return df['calg_ts'].apply(map_calg_clinstat)
    else: # None
        return df.apply(panss_OR_calg2clinstat, axis=1)
    
def get_visit_id_catie(visit):
    """
    Extract visit IDs from visit labels.
    
    Parameters:
    visit (list): List of visit labels.
    
    Returns:
    list: List of visit IDs.
    """
    y = [0 for _ in range(len(visit))]
    for idx,v in enumerate(visit):
        if 'Visit' in v and "FollowUp" not in v and "Phase" not in v:
            y[idx] = int(v.split("Visit")[1])
        elif v in ['Baseline', 'Screening']:
            y[idx] = 0
        else:
            y[idx] = y[idx-1] if idx > 0 else 0
    return y

def find_missing_required_visits(visit_numbers, required_visits=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]):
    """
    Identifies the earliest visit number missing two required visits before it 
    and labels visits accordingly. Used in CATIE data processing.

    Args:
        visit_numbers (list): A list of integers representing the visit numbers.
        required_visits (list): A list of integers representing the required visit numbers.

    Returns:
        tuple: A tuple where the first element is the visit number missing two required visits
               and the second element is a list with labels (1 or 0).
    """
    # Track missing visits from the required visits
    missing_visits = [visit for visit in visit_numbers if visit not in required_visits]

    if len(missing_visits) < 2:
        if len(visit_numbers) < len(required_visits) - 1:
            # Dropped in a very early time, label all visits as 0 except for the last one
            labels = [0] * (len(visit_numbers) - 1) + [1] if visit_numbers else [0]
            return visit_numbers[-1], labels
        
        # Not enough missing required visits
        labels = [0 for _ in range(len(visit_numbers))]
        return None, labels

    # Find the least number missing two required visits before
    target_missing = missing_visits[1]  # Second missing visit
    if target_missing > 0:
        target_missing -= 1  # Identify the previous visit to be the dropout visit
    
    # Labeling
    labels = []
    for visit in visit_numbers:
        labels.append(1 if visit >= target_missing else 0)

    return target_missing, labels

# -----------------------------------------------------------------------------
##  Following functions are used in survival_analysis.py, visualization part
# -----------------------------------------------------------------------------
def GetSMK(smoker_yn_1):
    if smoker_yn_1=='non-smoker':
        return 0
    if smoker_yn_1=='smoker':
        return 2
    if smoker_yn_1=='changer':
        return 1
    else:
        return 2

def GetClinstat2(clinstat_1):
    if clinstat_1==0:
        return 0
    if clinstat_1 in [1,2]:
        return 1
    
def bmistatus2cat(bmi_status):
    if bmi_status in ('nan','bmi1'):
        return 1
    if bmi_status=='bmi2':
        return 2
    if bmi_status=='bmi3':
        return 3
    if bmi_status=='bmi456':
        return 4
    if bmi_status=='transformer':
        return -1
    if bmi_status=='no_bmi':
        return -999
    
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
        df['smoker_yn'] = df['smoker_yn_1'].apply(GetSMK)
        df['smoker_yn_c'] = pd.Categorical(df['smoker_yn'], categories=[0, 1, 2], ordered=True)
        
    if 'clinstat_1' in df.columns:
        df['clinstat_2'] = df['clinstat_1'].apply(GetClinstat2)
        df['clinstat_2_c'] = pd.Categorical(df['clinstat_2'], categories=[0, 1], ordered=True)
        
    if 'bmi_status' in df.columns:
        df['bmi_cat'] = df['bmi_status'].apply(bmistatus2cat)

    return df

def plot_km_smoking_clinstat_curves(df_stepBD, df_catie, highlight_times=[180, 360, 540, 720, 900, 1080], filename="Figure_2.png"):
    """
    Plot Kaplan-Meier curves for smokers vs non-smokers stratified by clinical status.
    Parameters:
    df_stepBD (DataFrame): StepBD dataset
    df_catie (DataFrame): CATIE dataset
    highlight_times (list, optional): Times to highlight on the curve
    filename (str, optional): Filename to save the plot
    Returns:
    Median survival times for both datasets in each subgroup.
    """
    # Create a figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Adjust the size as needed
    axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate

    # Iterate over each dataframe and corresponding axis
    median_days = {}
    for df_t, ax in zip([df_stepBD, df_catie], axes):
        
        df = df_t.copy()
        df['clinstat_2_c'] = pd.Categorical(df['clinstat_2'], categories=[0, 1], ordered=True)
        df['weight'] = 1 / pow(df.groupby('subjectkey')['subjectkey'].transform('count'),1) # weighted by number of visits per subject
        ind = np.where(axes==ax)[0][0]
        print(["StepBD:", "CATIE:"][ind])
        
        # Create Kaplan-Meier objects for non-smokers, smokers, and excluders
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
        
        # Output median survival times for each subgroup
        median_well_pwds = kmf_s0.median_survival_time_
        median_unwell_pwds = kmf_s1.median_survival_time_
        median_well_pws = kmf_ns0.median_survival_time_
        median_unwell_pws = kmf_ns1.median_survival_time_
        print(f"Median survival time for Well PWDS: {median_well_pwds:.2f} days")
        print(f"Median survival time for Unwell PWDS: {median_unwell_pwds:.2f} days")
        print(f"Median survival time for Well PWS: {median_well_pws:.2f} days")
        print(f"Median survival time for Unwell PWS: {median_unwell_pws:.2f} days")

        # Plot the survival curves
        kmf_s0.plot(ax=ax, ci_alpha=0)
        kmf_s1.plot(ax=ax, ci_alpha=0)
        kmf_ns0.plot(ax=ax, ci_alpha=0)
        kmf_ns1.plot(ax=ax, ci_alpha=0)

        # Highlight specific time points
        # Disabled for a concise visualization
        # Set titles and labels
        if ind==0:
            ax.set_title('StepBD')
            median_days['StepBD'] = {
                'Well PWDS': median_well_pwds,
                'Unwell PWDS': median_unwell_pwds,
                'Well PWS': median_well_pws,
                'Unwell PWS': median_unwell_pws
            }
            
        if ind==1:
            ax.set_title('CATIE')
            ax.set_xlim(0, 550)
            ax.set_ylim(0, 1)
            median_days['CATIE'] = {
                'Well PWDS': median_well_pwds,
                'Unwell PWDS': median_unwell_pwds,
                'Well PWS': median_well_pws,
                'Unwell PWS': median_unwell_pws
            }
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    return median_days

def plot_km_bmi_clinstat_curves(df_stepBD, df_catie, highlight_times= [180, 360, 540, 720, 900, 1080], filename="Figure_3.png"):
    """
    Plot Kaplan-Meier curves for different BMI categories stratified by clinical status.
    
    Parameters:
    df_stepBD (DataFrame): StepBD dataset
    df_catie (DataFrame): CATIE dataset
    highlight_times (list, optional): Times to highlight on the curve
    filename (str, optional): Filename to save the plot
    Returns:
    Median survival times for both datasets in each subgroup.
    """
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()
    median_days = {}
    for df_t, ax in zip([df_stepBD, df_catie], axes):
        df = df_t.copy()
        df['clinstat_2_c'] = pd.Categorical(df['clinstat_2'], categories=[0, 1], ordered=True)
        ind = np.where(axes==ax)[0][0]
        print(["StepBD:", "CATIE:"][ind])
        
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
        
        # Output median survival times for each subgroup
        median_well_bmi2 = kmf_sb2.median_survival_time_
        median_well_bmi3 = kmf_sb3.median_survival_time_
        median_well_bmi456 = kmf_sb456.median_survival_time_
        median_unwell_bmi2 = kmf_nsb2.median_survival_time_
        median_unwell_bmi3 = kmf_nsb3.median_survival_time_
        median_unwell_bmi456 = kmf_nsb456.median_survival_time_
        print(f"Median survival time for Well BMI 18<=BMI<25: {median_well_bmi2:.2f} days")
        print(f"Median survival time for Well BMI 25<=BMI<30: {median_well_bmi3:.2f} days")
        print(f"Median survival time for Well BMI>=30: {median_well_bmi456:.2f} days")
        print(f"Median survival time for Unwell BMI 18<=BMI<25: {median_unwell_bmi2:.2f} days")
        print(f"Median survival time for Unwell BMI 25<=BMI<30: {median_unwell_bmi3:.2f} days")
        print(f"Median survival time for Unwell BMI>=30: {median_unwell_bmi456:.2f} days")

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
            median_days['StepBD'] = {
                'Well 18<=BMI<25': median_well_bmi2,
                'Well 25<=BMI<30': median_well_bmi3,
                'Well BMI>=30': median_well_bmi456,
                'Unwell 18<=BMI<25': median_unwell_bmi2,
                'Unwell 25<=BMI<30': median_unwell_bmi3,
                'Unwell BMI>=30': median_unwell_bmi456
            }
        if ind == 1:
            ax.set_title('CATIE')
            ax.set_xlim(0, 550)
            ax.set_ylim(0.3, 1)
            median_days['CATIE'] = {
                'Well 18<=BMI<25': median_well_bmi2,
                'Well 25<=BMI<30': median_well_bmi3,
                'Well BMI>=30': median_well_bmi456,
                'Unwell 18<=BMI<25': median_unwell_bmi2,
                'Unwell 25<=BMI<30': median_unwell_bmi3,
                'Unwell BMI>=30': median_unwell_bmi456
            }
            
        ax.set_xlabel('Days since first visit')
        ax.set_ylabel('Stay Probability')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    return median_days

# -----------------------------------------------------------------------------
##  Following functions are used in survival_analysis.py, Cox regression part
# -----------------------------------------------------------------------------
def _extract_row_from_model(model, var_name):
    """Grab HR, 95% CI, z, p for the single comparison term."""
    s = model.summary
    idx = var_name if var_name in s.index else f"{var_name}[T.1]"
    row = s.loc[idx]
    return {
        "HR": row["exp(coef)"],
        "HR, lower 95%": row["exp(coef) lower 95%"],
        "HR, upper 95%": row["exp(coef) upper 95%"],
        "z": row["z"],
        "Uncorrected p": row["p"],
    }

def _nobs_for_comparison(df, comp_name, ref_params, comp_params):
    """
    Count observations used in the *non-weighted* pairwise fit.
    We reuse run_pairwise_cox_model (weights_col=None) and look at the filtered df.
    """
    _, df_filtered = run_pairwise_cox_model(
        df, comparison_type=comp_name,
        ref_group_params=ref_params,
        comp_group_params=comp_params,
        weights_col=None
    )
    return len(df_filtered)

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
    if weighted, returns the number of observations in the comparison group;
    else returns a fitted CoxPHFitter and a DataFrame with the comparison column.
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
        {'smoker_yn_c': 0, 'clinstat_2_c': 0},  # Well non-smoker
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
        {'smoker_yn_c': 0, 'clinstat_2_c': 0},  # Well non-smoker
        {'clinstat_2_c': 1},  # Unwell
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
        {'smoker_yn_c': 0, 'clinstat_2_c': 0},  # Well non-smoker
        {'smoker_yn_c': 2, 'clinstat_2_c': 1},  # Unwell smoker
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
        {'bmi_cat': 2, 'clinstat_2_c': 0},  # Well normal weight
        {'bmi_cat': 3},  # Overweight
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
        {'bmi_cat': 2, 'clinstat_2_c': 0},  # Well normal weight
        {'bmi_cat': 4},  # Obese
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
        {'bmi_cat': 2, 'clinstat_2_c': 0},  # Well normal weight
        {'clinstat_2_c': 1},  # Unwell
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
        {'bmi_cat': 2, 'clinstat_2_c': 0},  # Well normal weight
        {'bmi_cat': 3, 'clinstat_2_c': 1},  # Unwell overweight
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
        {'bmi_cat': 2, 'clinstat_2_c': 0},  # Well normal weight
        {'bmi_cat': 4, 'clinstat_2_c': 1},  # Unwell obese
        weights_col
    )
    results['obuw_vs_nw'] = {
        'model': obuw_vs_nw_cph,
        'p_value': obuw_vs_nw_cph.summary.loc['obuw_vs_nw[T.1]', 'p']
    }
    
    return results

# ---------- Excel-format helpers ----------
def _fmt_n(x):
    """34,856 style for counts; blank if NaN."""
    if pd.isna(x):
        return ""
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)

def _fmt_2f(x):
    """Fixed 2 decimals for HR, CI, z; blank if NaN."""
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

def _fmt_p(x):
    """
    p-values: if p < 0.005 -> '<0.005'; else 2 decimals (0.02, 0.27, ...).
    Blank if NaN.
    """
    if pd.isna(x):
        return ""
    try:
        p = float(x)
    except Exception:
        # if something odd slipped in, just show it
        return str(x)
    return "<0.005" if p < 0.005 else f"{p:.3f}"

def format_table_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a *string-formatted* copy ready to write to Excel."""
    out = df.copy()

    # 1) counts with thousands separator
    if "Number of Observations" in out:
        out["Number of Observations"] = out["Number of Observations"].map(_fmt_n)

    # 2) HR/CI/z to .2f
    for col in ["HR", "HR, lower 95%", "HR, upper 95%", "z"]:
        if col in out:
            out[col] = out[col].map(_fmt_2f)

    # 3) p-values with <0.005 rule, else .2f
    for col in ["Uncorrected p", "FDR-corrected p"]:
        if col in out:
            out[col] = out[col].map(_fmt_p)

    return out

def save_table_to_excel(df: pd.DataFrame, path: str, sheet_name: str = "Table"):
    """
    Write a table to Excel with the required *display* formatting
    (values already converted to strings). Sets reasonable column widths.
    """
    df_fmt = format_table_for_excel(df)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df_fmt.to_excel(writer, index=False, sheet_name=sheet_name)

        ws = writer.sheets[sheet_name]
        # autosize columns based on displayed strings
        for i, col in enumerate(df_fmt.columns):
            max_len = max([len(str(col))] + [len(str(v)) for v in df_fmt[col].tolist()])
            ws.set_column(i, i, min(45, max(10, max_len + 2)))

# --------- TABLE 2: Smoking × Illness severity ---------
def make_table2(df_stepbd, df_catie, out_csv="table2.csv", out_excel="table2.xlsx"):
    """
    Build Table 2 across both datasets:
    Rows:
      - PWS vs. well PWDS
      - Unwell vs. well PWDS
      - Unwell PWS vs. well PWDS
    """
    # 1) define comparisons: key -> (display label, ref_params, comp_params, var_name)
    # ref is "well PWDS": clinstat_2_c=0 and nonsmoker (0)
    cmp_defs = {
        "s_vs_nsw":      ("PWS vs. well PWDS",           {"smoker_yn_c": 0, "clinstat_2_c": 0}, {"smoker_yn_c": 2},                  "s_vs_nsw"),
        "uw_vs_nsw":     ("Unwell vs. well PWDS",        {"smoker_yn_c": 0, "clinstat_2_c": 0}, {"clinstat_2_c": 1},                  "uw_vs_nsw"),
        "suw_vs_nsw":    ("Unwell PWS vs. well PWDS",    {"smoker_yn_c": 0, "clinstat_2_c": 0}, {"smoker_yn_c": 2, "clinstat_2_c": 1},"suw_vs_nsw"),
    }

    # 2) run weighted pairwise fits to get HR / z / p
    step_w = run_smoking_pairwise_comparisons(df_stepbd, with_weights=True)
    catie_w = run_smoking_pairwise_comparisons(df_catie, with_weights=True)

    # 3) build rows for both datasets
    rows = []
    pvals_for_fdr = []

    def add_block(df_raw, weighted_models, dataset_name):
        for key, (label, refp, compp, varname) in cmp_defs.items():
            # N from NON-weighted filter
            nobs = _nobs_for_comparison(df_raw, key, refp, compp)

            # weighted results
            model = weighted_models[key]["model"]
            stats = _extract_row_from_model(model, varname)
            pvals_for_fdr.append(stats["Uncorrected p"])

            rows.append({
                "Dataset": dataset_name,
                "Comparison": label,
                "Number of Observations": nobs,
                **stats,  # HR, CI, z, uncorrected p
            })

    add_block(df_stepbd, step_w, "StepBD")
    add_block(df_catie,  catie_w, "CATIE")

    # 4) FDR across all 6 rows (StepBD + CATIE)
    _, p_adj = fdrcorrection(pvals_for_fdr, alpha=0.05)
    for r, p in zip(rows, p_adj):
        r["FDR-corrected p"] = p

    # 5) finalize table (order & save)
    table2 = pd.DataFrame(rows)[[
        "Dataset", "Comparison", "Number of Observations",
        "HR", "HR, lower 95%", "HR, upper 95%", "z", "Uncorrected p", "FDR-corrected p"
    ]]

    # optional sort to match your screenshot order
    table2["order"] = table2["Dataset"].map({"StepBD": 0, "CATIE": 1}) + \
                      table2["Comparison"].map({
                          "PWS vs. well PWDS": 0,
                          "Unwell vs. well PWDS": 1,
                          "Unwell PWS vs. well PWDS": 2
                      })/10.0
    table2 = table2.sort_values(["order"]).drop(columns=["order"]).reset_index(drop=True)

    table2.to_csv(out_csv, index=False)
    save_table_to_excel(table2, out_excel, sheet_name="Table 2")
    return table2

# --------- TABLE 3: BMI × Illness severity ---------
def make_table3(df_stepbd, df_catie, out_csv="table3.csv", out_excel="table3.xlsx"):
    """
    Build Table 3 across both datasets:
      - Overweight vs. well normal weight
      - Obese vs. well normal weight
      - Unwell vs. well normal weight
      - Unwell overweight vs. well normal weight
      - Unwell obese vs. well normal weight
    """
    # mapping: key -> (label, ref_params, comp_params, var_name)
    # ref is "well normal weight": bmi_cat=2 and clinstat_2_c=0
    cmp_defs = {
        "o_vs_nw":     ("Overweight vs. well normal weight",   {"bmi_cat": 2, "clinstat_2_c": 0}, {"bmi_cat": 3},                      "o_vs_nw"),
        "obe_vs_nw":  ("Obese vs. well normal weight",         {"bmi_cat": 2, "clinstat_2_c": 0}, {"bmi_cat": 4},                      "obe_vs_nw"),
        "uw_vs_nw":   ("Unwell vs. well normal weight",        {"bmi_cat": 2, "clinstat_2_c": 0}, {"clinstat_2_c": 1},                      "uw_vs_nw"),
        "ouw_vs_nw":  ("Unwell overweight vs. well normal weight", {"bmi_cat": 2, "clinstat_2_c": 0}, {"bmi_cat": 3, "clinstat_2_c": 1}, "ouw_vs_nw"),
        "obuw_vs_nw": ("Unwell obese vs. well normal weight",  {"bmi_cat": 2, "clinstat_2_c": 0}, {"bmi_cat": 4, "clinstat_2_c": 1},    "obuw_vs_nw"),
    }

    # weighted models (HR/CI/z/p)
    step_w = run_bmi_pairwise_comparisons(df_stepbd, with_weights=True)
    catie_w = run_bmi_pairwise_comparisons(df_catie, with_weights=True)

    rows = []
    pvals_for_fdr = []

    def add_block(df_raw, weighted_models, dataset_name):
        for key, (label, refp, compp, varname) in cmp_defs.items():
            nobs = _nobs_for_comparison(df_raw, key, refp, compp)
            model = weighted_models[key]["model"]
            stats = _extract_row_from_model(model, varname)
            pvals_for_fdr.append(stats["Uncorrected p"])
            rows.append({
                "Dataset": dataset_name,
                "Comparison": label,
                "Number of Observations": nobs,
                **stats,
            })

    add_block(df_stepbd, step_w, "StepBD")
    add_block(df_catie,  catie_w, "CATIE")

    # FDR across all 10 rows
    _, p_adj = fdrcorrection(pvals_for_fdr, alpha=0.05)
    for r, p in zip(rows, p_adj):
        r["FDR-corrected p"] = p

    table3 = pd.DataFrame(rows)[[
        "Dataset", "Comparison", "Number of Observations",
        "HR", "HR, lower 95%", "HR, upper 95%", "z", "Uncorrected p", "FDR-corrected p"
    ]]

    # sort to match your screenshot order
    order_map = {
        "Overweight vs. well normal weight": 0,
        "Obese vs. well normal weight": 1,
        "Unwell vs. well normal weight": 2,
        "Unwell overweight vs. well normal weight": 3,
        "Unwell obese vs. well normal weight": 4,
    }
    table3["order"] = table3["Dataset"].map({"StepBD": 0, "CATIE": 1}) + \
                      table3["Comparison"].map(order_map)/10.0
    table3 = table3.sort_values(["order"]).drop(columns=["order"]).reset_index(drop=True)

    table3.to_csv(out_csv, index=False)
    save_table_to_excel(table3, out_excel, sheet_name="Table 3")
    return table3

# --------- Example usage (after you’ve prepared clean StepBD/CATIE dataframes) ---------
# table2 = make_table2(df_stepbd, df_catie)
# table3 = make_table3(df_stepbd, df_catie)
# print(table2)
# print(table3)