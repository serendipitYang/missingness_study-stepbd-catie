import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from stepbd_processing_utils import (
    locf_imputation, 
    select_least_nan_row, 
    map_clinstat, 
    get_event, 
    get_visit_id, 
    get_event_by_visit,
    compute_schedule_visit
)

def count_bmi_category_changes(df, column='obese'):
    """
    Count the changes in BMI categories for individuals.
    
    Parameters:
    df (DataFrame): Input DataFrame.
    column (str, optional): Column to analyze changes. Defaults to 'obese'.
    
    Returns:
    tuple: Counts of changes from 0 to 1 and 1 to 0.
    """
    def count_changes(value_list):
        """
        Counts the changes from 0 to 1 and from 1 to 0 in a list.
        """
        changes_0_to_1 = 0
        changes_1_to_0 = 0

        # Iterate through the list, except the last item to avoid index out of range
        for i in range(len(value_list) - 1):
            # Check for change from 0 to 1
            if value_list[i] == 0 and value_list[i + 1] == 1:
                changes_0_to_1 += 1
            # Check for change from 1 to 0
            elif value_list[i] == 1 and value_list[i + 1] == 0:
                changes_1_to_0 += 1

        return changes_0_to_1, changes_1_to_0
    
    return count_changes

def preprocess_bmi_data(input_file, output_dir):
    """
    Preprocess StepBD BMI data.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Directory to save processed data.
    
    Returns:
    DataFrame: Processed BMI data.
    """
    # Read the raw data
    df_all = pd.read_csv(input_file)
    
    # Select relevant columns for BMI analysis
    df_bmi = df_all[['subjectkey', 'src_subject_id', 'interview_age', 'daysrz', 
                     'bmi_cat', 'obese', 'height_std', 'weight_std',
                     'ppd', 'smoker_yn', 'madrstot', 'ymrstot', 'visit', 'clinstat']]
    
    # Impute BMI category
    columns_to_impute = ['bmi_cat']
    df_imputed1 = locf_imputation(df_bmi, columns_to_impute)
    df_imputed1 = df_imputed1.dropna(subset=['bmi_cat'])
    df_imputed1['bmi_cat'] = df_imputed1['bmi_cat'].astype('int')
    
    # Impute clinical status
    df_imputed2 = locf_imputation(df_imputed1, ['clinstat'])
    df_imputed2['clinstat'] = df_imputed2['clinstat'].astype('int')
    
    # Categorize BMI groups
    bmi1, bmi2, bmi3, bmi456, transformer = [], [], [], [], []
    for k in tqdm(list(set(df_imputed2['subjectkey']))):
        bmi_cates = list(df_imputed2[df_imputed2['subjectkey']==k]['bmi_cat'])
        all_bmi_cates = [int(s) for s in list(set(bmi_cates))]
        
        if len(set(bmi_cates)) != 1:
            transformer.append(k)
            continue
        
        if all_bmi_cates[0] == 1:
            bmi1.append(k)
        elif all_bmi_cates[0] == 2:
            bmi2.append(k)
        elif all_bmi_cates[0] == 3:
            bmi3.append(k)
        elif all_bmi_cates[0] >= 4:
            bmi456.append(k)
    
    # Add BMI status labels
    bmi_status = [''] * len(df_imputed2)
    for idx, bmi in enumerate(df_imputed2['subjectkey']):
        if bmi in bmi2:
            bmi_status[idx] = 'bmi2'
        if bmi in bmi3:
            bmi_status[idx] = 'bmi3'
        if bmi in bmi456:
            bmi_status[idx] = 'bmi456'
        if bmi in transformer:
            bmi_status[idx] = 'transformer'
    df_imputed2['bmi_status'] = bmi_status
    
    # Load MADRS and YMRS data
    df_madrs = pd.read_csv(os.path.join(os.path.dirname(input_file), 
                                        'STEP-BD data/Text files from NDA/madrs01.txt'), 
                           header=1, delimiter=r"\s+")
    df_ymrs = pd.read_csv(os.path.join(os.path.dirname(input_file), 
                                       'STEP-BD data/Text files from NDA/ymrs01.txt'), 
                          header=1, delimiter=r"\s+")
    
    # Rename columns
    rename_dict = {
        "The NDAR Global Unique Identifier (GUID) for research subject": "subjectkey",
        "days since randomization": "daysrz"
    }
    df_madrs = df_madrs.rename(columns=rename_dict)
    df_ymrs = df_ymrs.rename(columns=rename_dict)
    
    # Add indicator columns
    df_madrs['is_madrs'] = 1
    df_ymrs['is_ymrs'] = 1
    
    # Merge with MADRS and YMRS data
    df_bmi_2 = df_imputed2.merge(
        df_madrs[['subjectkey', 'daysrz', 'is_madrs']], 
        on=['subjectkey', 'daysrz'], 
        how='left'
    )
    df_bmi_3 = df_bmi_2.merge(
        df_ymrs[['subjectkey', 'daysrz', 'is_ymrs']], 
        on=['subjectkey', 'daysrz'], 
        how='left'
    )
    df_bmi_3 = df_bmi_3.sort_values(by=['subjectkey', 'daysrz'], ascending=True)
    
    # Drop duplicates
    grouped = df_bmi_3.groupby(['subjectkey', 'daysrz'], as_index=False)
    df_bmi_4 = grouped.apply(select_least_nan_row).reset_index(drop=True)
    df_bmi_4 = df_bmi_4.sort_values(by=['subjectkey', 'daysrz'], ascending=True)
    
    # Label scheduled visits
    is_scheduled = [0] * len(df_bmi_4)
    for idx, ism in enumerate(tqdm(df_bmi_4['is_madrs'])):
        if not (np.isnan(ism) and np.isnan(df_bmi_4['is_ymrs'][idx])):
            is_scheduled[idx] = 1
    df_bmi_4['is_scheduled'] = is_scheduled
    
    # Map clinical status
    df_bmi_4['clinstat'] = df_bmi_4['clinstat'].astype('int')
    df_bmi_4['clinstat_1'] = df_bmi_4['clinstat'].apply(map_clinstat)
    
    # Compute scheduled visits
    df_bmi_5 = compute_schedule_visit(df_bmi_4)
    
    # Prepare final dataset for survival analysis
    df_new = pd.DataFrame([], columns=list(df_bmi_5.columns)+['event_occurs'])
    for idx, subject in enumerate(tqdm(list(set(df_bmi_5['subjectkey'])))):
        subject_df = df_bmi_5[df_bmi_5['subjectkey'] == subject]
        subject_df['event_occurs'] = get_event(list(subject_df['is_scheduled']))
        
        # Keep only the first event occurrence row
        if 1 in subject_df['event_occurs']:
            subject_df1 = pd.concat([
                subject_df[subject_df['event_occurs'] == 0],
                subject_df[subject_df['event_occurs'] == 1].head(1)
            ], axis=0)
        else:
            subject_df1 = subject_df
        
        df_new = pd.concat([df_new, subject_df1], axis=0)
    
    # Ensure only positive days are included
    df_new = df_new[df_new['daysrz'] >= 0].reset_index(drop=True)
    
    return df_new

def process_bmi_data(input_file, output_file):
    """
    Process and save BMI data for survival analysis.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the processed data.
    """
    # Process the data
    df_processed = preprocess_bmi_data(input_file, output_dir=os.path.dirname(output_file))
    
    # Save the processed data
    df_processed.to_csv(output_file, index=False)
    print(f"Processed BMI data saved to {output_file}")
    
    # Print basic statistics
    print("\nData Processing Statistics:")
    print("Total subjects:", len(set(df_processed['subjectkey'])))
    print("BMI status distribution:")
    print(Counter(df_processed['bmi_status']))
    print("\nEvent occurrence distribution:")
    print(Counter(df_processed['event_occurs']))

def main():
    """
    Main function to run the BMI data processing.
    """
    # Paths - replace these with your actual paths
    input_file = '../STEP_BD_pipelined/step_bd_final_data_debugged_dedupe_new_vars_med_bin_derived_complete.csv'
    output_file = '../data4survivals/stepBD_2missing_bmi_w_impute.csv'
    
    # Process the data
    process_bmi_data(input_file, output_file)

if __name__ == "__main__":
    main()
