import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

def locf_imputation(df, columns):
    """
    Impute missing values using Last Observation Carried Forward (LOCF) method,
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

def compute_schedule_visit(df, time_window=15):
    """
    Compute scheduled visits based on predefined visit schedule.
    
    Parameters:
    df (DataFrame): Input DataFrame.
    time_window (int, optional): Time window for matching visits. Defaults to 15.
    
    Returns:
    DataFrame: DataFrame with schedule_visit column added.
    """
    # Define the visit schedule in terms of days
    visit_schedule = {
        "Baseline": 0,
        "Visit1": 60,
        "Visit2": 120,
        "Visit3": 180,
        "Visit4": 240,
        "Visit5": 300,
        "Visit6": 360,
        "Visit7": 420,
        "Visit8": 480,
        "Visit9": 540,
        "Visit10": 600,
        "Visit11": 660,
        "Visit12": 720,
        "Visit13": 780,
        "Visit14": 840,
        "Visit15": 900,
        "Visit16": 960,
        "Visit17": 1020,
        "Visit18": 1080
    }
    
    # Initialize a new column for schedule_visit with 'missed'
    df['schedule_visit'] = 'missed'
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Check if daysrz falls within the window of any scheduled visit
        for visit, day in visit_schedule.items():
            if abs(row['daysrz'] - day) <= time_window:
                df.at[index, 'schedule_visit'] = visit
                break  # Assign only the first matching visit
    
    return df

def map_clinstat(df, is_panss=True):
    """
    Map clinical status based on PANSS total or Calgary score.
    
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

    def panss_or_calg_clinstat(row):
        pt = row.get('clinstat_pt', np.nan)
        cg = row.get('clinstat_cg', np.nan)
        
        if np.isnan(pt) and np.isnan(cg):
            return np.nan
        if pd.notna(pt) and pt == 1:
            return 1
        if pd.notna(cg) and cg == 1:
            return 1
        return 0

    if is_panss:
        return df['panss_total'].apply(map_panss_clinstat)
    elif is_panss is False:
        return df['calg_ts'].apply(map_calg_clinstat)
    else:
        return df.apply(panss_or_calg_clinstat, axis=1)

def get_visit_id(visit):
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
    and labels visits accordingly.

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
