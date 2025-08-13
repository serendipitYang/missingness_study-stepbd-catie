import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import warnings

# Import utility functions from local utils.py
from utils import (
    locf_imputation, 
    panss_cols,
    clgry_cols,
    cig_cols,
    map_clinstat_panss_or_calg,
    get_visit_id_catie, # Different ID system of CATIE from StepBD
    find_missing_required_visits
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

def preprocess_catie_smoking_data(input_file, output_dir, demo_file, panss_file, clgry_file, cig_file):
    """
    Preprocess CATIE smoking data.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Directory to save processed data.
    demo_file (str): Path to the demographics data file.
    panss_file (str): Path to the PANSS data file.
    clgry_file (str): Path to the Calgary data file.
    cig_file (str): Path to the cigarettes data file.

    Returns:
    DataFrame: Processed smoking data.
    """
    # Read visit data
    logger.info("Reading visit data...")
    df_all = pd.read_csv(input_file, skiprows=1, names=[
        'CATIEID', 'subjectkey', 'gender', 'visit', 'visitid', 'TRUNCVIS',
        'PHASE', 'MISSED', 'ATTENDED', 'DISCON', 'VISDAY'
    ])

    # Read essential datasets, with selected columns
    df_panss = pd.read_csv(panss_file, skiprows=2, delimiter="\t", names=panss_cols)
    df_clgry = pd.read_csv(clgry_file, skiprows=2, delimiter="\t", names=clgry_cols)
    df_cig = pd.read_csv(cig_file, skiprows=2, delimiter="\t", names=cig_cols)

    # Prepare data for merging
    convert_columns = ['subjectkey', 'visit', 'visitid']
    for df in [df_all, df_panss, df_clgry, df_cig]:
        for col in convert_columns:
            df[col] = df[col].astype('str')

    # Merge datasets, merge on subjectkey, visit, and visitid as join keys
    logger.info("Merging datasets...")
    df_all_1 = df_all.merge(df_panss[['subjectkey', 'visit', 'visitid',
            'panss_general', 'panss_negative','panss_positive', 'panss_total']], 
                                        on=['subjectkey', 'visit', 'visitid'], how='left')
    df_all_2 = df_all_1.merge(df_clgry[['subjectkey', 'visit', 'visitid',
            'calg_ts']], on=['subjectkey', 'visit', 'visitid'], how='left')
    df_all_3 = df_all_2.merge(df_cig[['subjectkey', 'visit', 'visitid',
            'cs06', 'cscigs']], on=['subjectkey', 'visit', 'visitid'], how='left')
    # Rename columns
    logger.info("Renaming columns...")
    df_smoking=df_all_3.rename(columns={
        'VISDAY': 'daysrz',
        'cs06': 'smoker_yn',
        'cscigs': 'cigarettes_past_7_days'
    })
    
    # Clinical status determination
    logger.info("Determining clinical status by PANSS or Calgary score...")
    df_smoking['clinstat_pt'] = map_clinstat_panss_or_calg(df_smoking, is_panss=True)
    df_smoking['clinstat_cg'] = map_clinstat_panss_or_calg(df_smoking, is_panss=False)
    df_smoking['clinstat_ptORcg'] = map_clinstat_panss_or_calg(df_smoking, is_panss=None)
    # Impute clinical status
    logger.info("Imputing clinical status...")
    df_smoking['clinstat'] = df_smoking['clinstat_ptORcg'].copy()
    columns_to_impute = ['clinstat']
    df_smoking_imputed= locf_imputation(df_smoking, columns_to_impute)
    df_smoking_2 = df_smoking_imputed.dropna(subset=['clinstat'])
    df_smoking_2['clinstat_1'] = df_smoking_2['clinstat'].astype('int')

    # Determine smoking status by Screening or Baseline visit
    logger.info("Determining smoking status by Screening or Baseline visit...")
    smokers = []
    for idx_sk,sk in enumerate(tqdm(set(df_smoking_2['subjectkey']))):
        df_temp = df_smoking_2[(df_smoking_2['subjectkey']==sk) & (df_smoking_2['visit'].isin(['Screening','Baseline']))]
        if 1.0 in list(df_temp['smoker_yn']):
            smokers.append(sk)
    non_smokers = []
    for idx_sk,sk in enumerate(tqdm(set(df_smoking_2['subjectkey']))):
        df_temp = df_smoking_2[(df_smoking_2['subjectkey']==sk) & (df_smoking_2['visit'].isin(['Screening','Baseline']))]
        if 0.0 in list(df_temp['smoker_yn']):
            non_smokers.append(sk)
    missing_baseliners = list(set(df_smoking_2[(~df_smoking_2['subjectkey'].isin(list(set(smokers) | set(non_smokers)))) & (df_smoking_2['visit'].isin(['Screening','Baseline']))]['subjectkey']))
    logger.info(
        "Baseline classification: %d smokers, %d non-smokers, %d missing",
        len(smokers), len(non_smokers), len(missing_baseliners)
    )

    # Mark smoking status
    smoking_status = [''] * len(df_smoking_2)
    for idx, sk in enumerate(df_smoking_2['subjectkey']):
        if sk in smokers:
            smoking_status[idx] = 'smoker'
        elif sk in non_smokers:
            smoking_status[idx] = 'non-smoker'
        else:
            smoking_status[idx] = 'missing_baseliner'
    df_smoking_2['smoker_yn_1'] = smoking_status

    # Prepare for survival analysis
    df_smoking_filtered = df_smoking_2[
        df_smoking_2['visit'].str.match(r'^(Visit\d+|Baseline|Screening)$', na=False)
    ]

    # Prepare final dataset for survival analysis
    logger.info("Computing event occurrences for %d subjects...", len(set(df_smoking_filtered['subjectkey'])))
    df_new = pd.DataFrame([], columns=list(df_smoking_filtered.columns)+['visit_id', 'event_occurs'])
    for idx, subject in enumerate(tqdm(list(set(df_smoking_filtered['subjectkey'])))):
        subject_df = df_smoking_filtered[df_smoking_filtered['subjectkey'] == subject]
        visit_list = get_visit_id_catie(list(subject_df['visit']))
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

    logger.info("Filtering for first event occurrence per subject...")
    df_final = pd.DataFrame([], columns=df_new.columns)
    for idx,s in enumerate(tqdm(list(set(df_new['subjectkey'])))):
        df_temp = df_new[df_new['subjectkey']==s]
        first_one_index = df_temp.drop(df_temp[df_temp['event_occurs'].eq(1)].index[1:])
        df_final = pd.concat([df_final,first_one_index],axis=0)

    out_path = os.path.join(output_dir, 'CATIE_smoking_final_event_occurrence.csv')
    df_final.to_csv(out_path, index=False)
    logger.info("Final entries saved to %s (%d rows)", out_path, df_final.shape[0])
    return out_path, df_final

def process_catie_smoking_data(input_file, output_dir, demo_file, panss_file, clgry_file, cig_file):
    """
    Process and save CATIE smoking data for survival analysis.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Path to save the processed data file.
    demo_file (str): Path to the demographics data file.
    panss_file (str): Path to the PANSS data file.
    clgry_file (str): Path to the Calgary data file.
    cig_file (str): Path to the cigarettes data file.    
    """
    # Process the data
    out_path, df_processed = preprocess_catie_smoking_data(input_file, output_dir, demo_file, panss_file, clgry_file, cig_file)
    
    # Save the processed data
    print(f"Processed CATIE smoking data saved to {out_path}")
    
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
    demo_file = '../CATIE/CATIE data from NIH NDA/demo01.txt'
    panss_file = '../CATIE/CATIE data from NIH NDA/panss01.txt'
    clgry_file = '../CATIE/CATIE data from NIH NDA/clgry01.txt'
    cig_file = '../CATIE/CATIE data from NIH NDA/cgis01.txt'
    output_dir = '../data4survivals_1/'
    
    # Process the data
    process_catie_smoking_data(input_file, output_dir, demo_file, panss_file, clgry_file, cig_file)

if __name__ == "__main__":
    main()