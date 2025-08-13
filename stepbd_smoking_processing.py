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
    select_least_nan_row, 
    map_clinstat, 
    get_event, 
    get_visit_id, 
    get_event_by_visit,
    compute_schedule_visit,
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

def preprocess_smoking_data(input_file, baseline_file, output_dir, madrs_file=None, ymrs_file=None):
    """
    Preprocess StepBD smoking data.
    
    Parameters:
        input_file (str): Path to the input visit CSV file, ie. entry table.
        baseline_file(str): Path to the input baseline CSV file, ie. ade file.
        output_dir (str): Directory to save processed data.
        madrs_file (str): Path to MADRS text file.
        ymrs_file (str): Path to YMRS text file.
    
    Returns:
        DataFrame: Processed smoking data.
    """
    # Read the raw data
    df_all = pd.read_csv(input_file)
    df_ade = pd.read_csv(baseline_file, header=1, delimiter=r"\s+") # ade file
    
    # Label smokers. non_smokers, and missing_baseliner based on baseline visit
    smokers = set(df_ade[df_ade['nicotine packs per day']>0]['The NDAR Global Unique Identifier (GUID) for research subject'])
    non_smokers = set(df_ade[df_ade['nicotine packs per day']==0]['The NDAR Global Unique Identifier (GUID) for research subject'])
    missing_baseliner = set(df_ade[df_ade['nicotine packs per day'].isna()]['The NDAR Global Unique Identifier (GUID) for research subject'])
    # print(f"Number of \nsmokers: {len(smokers)};\nnon-smokers: {len(non_smokers)};\nmissing-baseliners: {len(missing_baseliner)}")
    logger.info(
        "Baseline classification: %d smokers, %d non-smokers, %d missing",
        len(smokers), len(non_smokers), len(missing_baseliner)
    )

    # Select relevant columns for smoking analysis
    df_smkornot = df_all[['subjectkey', 'src_subject_id', 'interview_age', 'daysrz', 
                          'smoker_yn', 'ppd', 'madrstot', 'ymrstot', 'visit', 'clinstat']]
    # Filter out subjects not in the baseline file
    logger.info("Filtering subjects based on baseline data...")
    df_smkornot = df_smkornot[df_smkornot['subjectkey'].isin(set(df_ade['The NDAR Global Unique Identifier (GUID) for research subject']))]

    # Impute clinical status
    logger.info("Imputing clinical status...")
    columns_to_impute = ['clinstat']
    df_smkornot_imputed = locf_imputation(df_smkornot, columns_to_impute)
    df_smkornot_1 = df_smkornot_imputed.dropna(subset=['clinstat'])
    
    # Smoking status settled down, no need to impute
    # Label smoking status in the entry table
    df_smkornot_2 = df_smkornot_1.copy()
    smoking_status = ['' for _ in range(len(df_smkornot_2))]
    for idx,sk in enumerate(df_smkornot_2['subjectkey']):
        if sk in smokers:
            smoking_status[idx] = 'smoker'
        elif sk in non_smokers:
            smoking_status[idx] = 'non-smoker'
        else:
            smoking_status[idx] = 'missing_baseliner'
        df_smkornot_2['smoker_yn_1'] = smoking_status
    
    # Load MADRS and YMRS data
    logger.info("Loading MADRS and YMRS files")
    df_madrs = pd.read_csv(madrs_file, header=1, delimiter=r"\s+")
    df_ymrs = pd.read_csv(ymrs_file, header=1, delimiter=r"\s+")
    
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
    
    logger.info(
        "After deduplication: %d rows across %d subjects",
        df_smkornot_5.shape[0], df_smkornot_5['subjectkey'].nunique()
    )

    # Label scheduled visits
    logger.info("Labeling scheduled visits, based on the validness of MADRS/YMRS record...")
    is_scheduled = [0 for _ in range(len(df_smkornot_5))]
    for idx, ism in enumerate(tqdm(df_smkornot_5['is_madrs'])):
        if not (np.isnan(ism) and np.isnan(df_smkornot_5['is_ymrs'][idx])):
            is_scheduled[idx] = 1
    df_smkornot_5['is_scheduled'] = is_scheduled
    logger.info(
        "Scheduled visits labeled: %d scheduled of %d total rows",
        sum(is_scheduled), len(df_smkornot_5)
    )
    
    # Map clinical status
    df_smkornot_5['clinstat'] = df_smkornot_5['clinstat'].astype('int')
    df_smkornot_5['clinstat_1'] = df_smkornot_5['clinstat'].apply(map_clinstat)
    
    # Compute scheduled visits
    df_smkornot_6 = compute_schedule_visit(df_smkornot_5)
    
    # Prepare final dataset for survival analysis
    subjects = list(set(df_smkornot_6['subjectkey']))
    logger.info("Computing event occurrences for %d subjects...", len(subjects))
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
    logger.info("Event assignment complete: resulting rows = %d", df_new.shape[0])

    # Ensure only the first event occurrence is kept per subject
    logger.info("Filtering for first event occurrence per subject...")
    df_final = pd.DataFrame([], columns=df_new.columns)
    for idx,s in enumerate(tqdm(list(set(df_new['subjectkey'])))):
        df_temp = df_new[df_new['subjectkey']==s]
        first_one_index = df_temp.drop(df_temp[df_temp['event_occurs'].eq(1)].index[1:])
        df_final = pd.concat([df_final,first_one_index],axis=0)
    # Final event occurrence file in need to for plotting and Cox regression
    out_path = os.path.join(output_dir, 'stepBD_smoking_final_event_occurrence.csv')
    df_final.to_csv(out_path, index=False)
    logger.info("Final entries saved to %s (%d rows)", out_path, df_final.shape[0])
    return out_path, df_final

def process_smoking_data(input_file, baseline_file, output_dir, madrs_file=None, ymrs_file=None):
    """
    Process and save smoking data for survival analysis.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the processed data.
    """
    # Process the data
    out_path, df_processed = preprocess_smoking_data(input_file, baseline_file, output_dir, madrs_file, ymrs_file)
    
    # Processed data saved, report now
    print(f"Processed smoking data saved to {out_path}")
    
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
    Configure your paths for input and output files here.
    """
    # Paths - replace these with your actual paths
    input_file = '../STEP_BD_pipelined/step_bd_final_data_debugged_dedupe_new_vars_med_bin_derived_complete.csv'
    baseline_file = '../STEP_BD_pipelined/ade01.txt'
    output_dir = '../data4survivals_1/'
    madrs_file  = '../STEP-BD data/Text files from NDA/madrs01.txt'
    ymrs_file  = '../STEP-BD data/Text files from NDA/ymrs01.txt'
    
    # Process the data
    process_smoking_data(input_file, baseline_file, output_dir, madrs_file, ymrs_file)

if __name__ == "__main__":
    main()
