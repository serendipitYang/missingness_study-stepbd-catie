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

def preprocess_smoking_data(input_file, output_dir):
    """
    Preprocess StepBD smoking data.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Directory to save processed data.
    
    Returns:
    DataFrame: Processed smoking data.
    """
    # Read the raw data
    df_all = pd.read_csv(input_file)
    
    # Select relevant columns for smoking analysis
    df_smkornot = df_all[['subjectkey', 'src_subject_id', 'interview_age', 'daysrz', 
                          'smoker_yn', 'ppd', 'madrstot', 'ymrstot', 'visit', 'clinstat']]
    
    # Impute clinical status
    columns_to_impute = ['clinstat']
    df_smkornot_imputed = locf_imputation(df_smkornot, columns_to_impute)
    df_smkornot_1 = df_smkornot_imputed.dropna(subset=['clinstat'])
    
    # Impute smoking status
    columns_to_impute = ['smoker_yn']
    df_smkornot_1_imputed = locf_imputation(df_smkornot_1, columns_to_impute)
    df_smkornot_2 = df_smkornot_1_imputed.dropna(subset=['smoker_yn'])
    df_smkornot_2['smoker_yn'] = df_smkornot_2['smoker_yn'].astype('int')
    
    # Categorize smoking status
    smokers, non_smokers, changers = [], [], []
    for k in tqdm(list(set(df_smkornot_2['subjectkey']))):
        smkstatus = list(df_smkornot_2[df_smkornot_2['subjectkey']==k]['smoker_yn'])
        all_smk_status = [int(s) for s in list(set(smkstatus))]
        
        if len(set(all_smk_status)) != 1:
            changers.append(k)
            continue
        
        if all_smk_status[0] == 0:
            non_smokers.append(k)
        elif all_smk_status[0] == 1:
            smokers.append(k)
    
    # Add smoking status labels
    smoking_status = [''] * len(df_smkornot_2)
    for idx, sk in enumerate(df_smkornot_2['subjectkey']):
        if sk in smokers:
            smoking_status[idx] = 'smoker'
        if sk in non_smokers:
            smoking_status[idx] = 'non-smoker'
        if sk in changers:
            smoking_status[idx] = 'changer'
    df_smkornot_2['smoker_yn_1'] = smoking_status
    
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
    df_smkornot_3 = df_smkornot_2.merge(
        df_madrs[['subjectkey', 'daysrz', 'is_madrs']], 
        on=['subjectkey', 'daysrz'], 
        how='left'
    )
    df_smkornot_4 = df_smkornot_3.merge(
        df_ymrs[['subjectkey', 'daysrz', 'is_ymrs']], 
        on=['subjectkey', 'daysrz'], 
        how='left'
    )
    df_smkornot_4 = df_smkornot_4.sort_values(by=['subjectkey', 'daysrz'], ascending=True)
    
    # Drop duplicates
    grouped = df_smkornot_4.groupby(['subjectkey', 'daysrz'], as_index=False)
    df_smkornot_5 = grouped.apply(select_least_nan_row).reset_index(drop=True)
    df_smkornot_5 = df_smkornot_5.sort_values(by=['subjectkey', 'daysrz'], ascending=True)
    
    # Label scheduled visits
    is_scheduled = [0] * len(df_smkornot_5)
    for idx, ism in enumerate(tqdm(df_smkornot_5['is_madrs'])):
        if not (np.isnan(ism) and np.isnan(df_smkornot_5['is_ymrs'][idx])):
            is_scheduled[idx] = 1
    df_smkornot_5['is_scheduled'] = is_scheduled
    
    # Map clinical status
    df_smkornot_5['clinstat'] = df_smkornot_5['clinstat'].astype('int')
    df_smkornot_5['clinstat_1'] = df_smkornot_5['clinstat'].apply(map_clinstat)
    
    # Compute scheduled visits
    df_smkornot_6 = compute_schedule_visit(df_smkornot_5)
    
    # Prepare final dataset for survival analysis
    df_new = pd.DataFrame([], columns=list(df_smkornot_6.columns)+['event_occurs'])
    for idx, subject in enumerate(tqdm(list(set(df_smkornot_6['subjectkey'])))):
        subject_df = df_smkornot_6[df_smkornot_6['subjectkey'] == subject]
        subject_df['event_occurs'] = get_event(list(subject_df['is_scheduled']))
        visit_list = get_visit_id(list(subject_df['visit']))
        subject_df['visit_id'] = visit_list
        subject_df['event_occurs'] = get_event_by_visit(visit_list)
        
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

def process_smoking_data(input_file, output_file):
    """
    Process and save smoking data for survival analysis.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the processed data.
    """
    # Process the data
    df_processed = preprocess_smoking_data(input_file, output_dir=os.path.dirname(output_file))
    
    # Save the processed data
    df_processed.to_csv(output_file, index=False)
    print(f"Processed smoking data saved to {output_file}")
    
    # Print basic statistics
    print("\nData Processing Statistics:")
    print("Total subjects:", len(set(df_processed['subjectkey'])))
    print("Smoking status distribution:")
    print(Counter(df_processed['smoker_yn_1']))
    print("\nEvent occurrence distribution:")
    print(Counter(df_processed['event_occurs']))

def main():
    """
    Main function to run the smoking data processing.
    """
    # Paths - replace these with your actual paths
    input_file = '../STEP_BD_pipelined/step_bd_final_data_debugged_dedupe_new_vars_med_bin_derived_complete.csv'
    output_file = '../data4survivals/stepBD_2missing_smoking_w_impute.csv'
    
    # Process the data
    process_smoking_data(input_file, output_file)

if __name__ == "__main__":
    main()
