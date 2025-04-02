import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from catie_processing_utils import (
    locf_imputation, 
    compute_schedule_visit, 
    map_clinstat,
    get_visit_id,
    find_missing_required_visits
)

def preprocess_catie_smoking_data(input_file, output_dir):
    """
    Preprocess CATIE smoking data.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Directory to save processed data.
    
    Returns:
    DataFrame: Processed smoking data.
    """
    # Read visit data
    df_all = pd.read_csv(input_file, skiprows=1, names=[
        'CATIEID', 'subjectkey', 'gender', 'visit', 'visitid', 'TRUNCVIS',
        'PHASE', 'MISSED', 'ATTENDED', 'DISCON', 'VISDAY'
    ])

    # Read essential datasets
    df_vitals = pd.read_csv(os.path.join(os.path.dirname(input_file), 'vitals01.txt'), delimiter="\t")
    df_panss = pd.read_csv(os.path.join(os.path.dirname(input_file), 'panss01.txt'), 
                            skiprows=2, 
                            delimiter="\t",
                            names=[
                                'collection_id', 'panss01_id', 'dataset_id', 
                                'subjectkey', 'visitid', 'visit', 
                                'panss_total', 'panss_positive', 'panss_negative'
                            ])
    df_clgry = pd.read_csv(os.path.join(os.path.dirname(input_file), 'clgry01.txt'), 
                            skiprows=2, 
                            delimiter="\t",
                            names=[
                                'collection_id', 'clgry01_id', 'dataset_id', 
                                'subjectkey', 'visitid', 'visit', 
                                'calg_ts'
                            ])
    df_cig = pd.read_csv(os.path.join(os.path.dirname(input_file), 'cgis01.txt'), 
                         skiprows=2, 
                         delimiter="\t",
                         names=[
                             'collection_id', 'cgis01_id', 'dataset_id', 
                             'subjectkey', 'visitid', 'visit', 
                             'cs06', 'cscigs'
                         ])

    # Prepare data for merging
    convert_columns = ['visitid', 'subjectkey']
    for df in [df_all, df_vitals, df_panss, df_clgry, df_cig]:
        for col in convert_columns:
            df[col] = df[col].astype('str')

    # Merge datasets
    df_smoking = df_all.merge(df_panss[['subjectkey', 'visit', 'visitid', 
                                         'panss_total', 'panss_positive', 'panss_negative']], 
                               on=['subjectkey', 'visit', 'visitid'], how='left')
    df_smoking = df_smoking.merge(df_clgry[['subjectkey', 'visit', 'visitid', 'calg_ts']], 
                                   on=['subjectkey', 'visit', 'visitid'], how='left')
    df_smoking = df_smoking.merge(df_cig[['subjectkey', 'visit', 'visitid', 'cs06', 'cscigs']], 
                                   on=['subjectkey', 'visit', 'visitid'], how='left')

    # Rename columns
    df_smoking.rename(columns={
        'VISDAY': 'daysrz',
        'cs06': 'smoker_yn',
        'cscigs': 'cigarettes_past_7_days'
    }, inplace=True)

    # Clinical status determination
    df_smoking['clinstat_pt'] = map_clinstat(df_smoking, is_panss=True)
    df_smoking['clinstat_cg'] = map_clinstat(df_smoking, is_panss=False)
    df_smoking['clinstat'] = map_clinstat(df_smoking, is_panss=None)

    # Impute clinical status
    columns_to_impute = ['clinstat']
    df_smoking_imputed = locf_imputation(df_smoking, columns_to_impute)
    df_smoking_imputed.dropna(subset=['clinstat'], inplace=True)
    df_smoking_imputed['clinstat_1'] = df_smoking_imputed['clinstat'].astype('int')

    # Impute smoker status
    columns_to_impute = ['smoker_yn']
    df_smoking_imputed = locf_imputation(df_smoking_imputed, columns_to_impute)
    df_smoking_imputed.dropna(subset=['smoker_yn'], inplace=True)
    df_smoking_imputed['smoker_yn'] = df_smoking_imputed['smoker_yn'].astype('int')

    # Categorize smokers
    smokers, non_smokers, changers = [], [], []
    for k in tqdm(list(set(df_smoking_imputed['subjectkey']))):
        smkstatus = list(df_smoking_imputed[df_smoking_imputed['subjectkey']==k]['smoker_yn'])
        all_smk_status = [int(s) for s in list(set(smkstatus))]
        
        if len(set(smkstatus)) != 1:
            changers.append(k)
            continue
        
        if all_smk_status[0] == 0:
            non_smokers.append(k)
        elif all_smk_status[0] == 1:
            smokers.append(k)

    # Mark smoking status
    smoking_status = [''] * len(df_smoking_imputed)
    for idx, sk in enumerate(df_smoking_imputed['subjectkey']):
        if sk in smokers:
            smoking_status[idx] = 'smoker'
        if sk in non_smokers:
            smoking_status[idx] = 'non-smoker'
        if sk in changers:
            smoking_status[idx] = 'changer'
    df_smoking_imputed['smoker_yn_1'] = smoking_status

    # Prepare for survival analysis
    df_smoking_filtered = df_smoking_imputed[
        df_smoking_imputed['visit'].str.match(r'^(Visit\d+|Baseline|Screening)$', na=False)
    ]

    # Prepare final dataset for survival analysis
    df_new = pd.DataFrame([], columns=list(df_smoking_filtered.columns)+['visit_id', 'event_occurs'])
    for idx, subject in enumerate(tqdm(list(set(df_smoking_filtered['subjectkey'])))):
        subject_df = df_smoking_filtered[df_smoking_filtered['subjectkey'] == subject]
        visit_list = get_visit_id(list(subject_df['visit']))
        subject_df['visit_id'] = visit_list
        target_missing, subject_df['event_occurs'] = find_missing_required_visits(visit_list)
        
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

def process_catie_smoking_data(input_file, output_file, missing_visits=2):
    """
    Process and save CATIE smoking data for survival analysis.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the processed data.
    missing_visits (int): Number of missing visits to define dropout.
    """
    # Process the data
    df_processed = preprocess_catie_smoking_data(input_file, output_dir=os.path.dirname(output_file))
    
    # Save the processed data
    df_processed.to_csv(output_file, index=False)
    print(f"Processed CATIE smoking data saved to {output_file}")
    
    # Print basic statistics
    print("\nData Processing Statistics:")
    print("Total subjects:", len(set(df_processed['subjectkey'])))
    print("Smoking status distribution:")
    print(Counter(df_processed['smoker_yn_1']))
    print("\nEvent occurrence distribution:")
    print(Counter(df_processed['event_occurs']))

def main():
    """
    Main function to run the CATIE smoking data processing.
    """
    # Paths - replace these with your actual paths
    input_file = '../CATIE/CATIE data from NIH NDA/VISIT.csv'
    output_file = '../data4survivals/CATIE_2missing_smoking_ptORcg_imputed.csv'
    
    # Process the data
    process_catie_smoking_data(input_file, output_file)

if __name__ == "__main__":
    main()
