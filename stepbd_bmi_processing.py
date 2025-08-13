import os
import logging, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

# Import utility functions from local utils.py
from utils import (
    bmi2cate,
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

def preprocess_stepbd_bmi_data(input_file, baseline_file, output_dir, madrs_file, ymrs_file):
    """
    Preprocess StepBD BMI data.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Directory to save processed data.
    baseline_file (str): Path to the baseline data file.
    madrs_file (str): Path to the MADRS data file.
    ymrs_file (str): Path to the YMRS data file.

    Returns:
    outpath (str): Path, with filename, to the saved processed data.
    DataFrame: Processed BMI data.
    """
    # Read the raw data
    df_all = pd.read_csv(input_file)
    df_ade = pd.read_csv(baseline_file, header=1, delimiter=r"\s+") # ade file
    
    # Select relevant columns for BMI analysis, only for subjects present in ADE file
    df_bmi = df_all[df_all['subjectkey'].isin(list(set(df_ade['The NDAR Global Unique Identifier (GUID) for research subject'])))][['subjectkey','src_subject_id','interview_age', 'daysrz', 'bmi', 'bmi_cat', 'obese', 'height_std', 'weight_std',
                 'ppd', 'smoker_yn', 'madrstot', 'ymrstot', 'visit', 'clinstat']]
    
    # Impute BMI value
    logger.info("Forward imputing BMI category...")
    columns_to_impute = ['bmi']
    df_imputed = locf_imputation(df_bmi, columns_to_impute)
    df_imputed1 = df_imputed.dropna(subset = ['bmi'])
    
    # Impute clinical status
    logger.info("Imputing clinical status category...")
    df_imputed2 = locf_imputation(df_bmi, ['clinstat'])
    df_imputed2['clinstat'] = df_imputed2['clinstat'].astype('int')
    
    # Categorize BMI groups, just use the baseline visit
    logger.info("Categorizing BMI groups based on the baseline visit...")
    df_bmi_1 = df_imputed2.copy()
    df_bmi_1_baseline = df_imputed1.drop_duplicates(subset = "subjectkey" ,keep='first')
    df_bmi_1_baseline['bmi_category'] = df_bmi_1_baseline['bmi'].apply(bmi2cate)
    print(Counter(df_bmi_1_baseline['bmi_category']))

    bmi1, bmi2, bmi3, bmi456, no_bmi_0 = [], [], [], [], []
    for k in tqdm(list(set(df_bmi_1_baseline['subjectkey']))):
        bmi_cates = list(df_bmi_1_baseline[df_bmi_1_baseline['subjectkey']==k]['bmi_category'])
        all_bmi_cates = [s for s in list(set(bmi_cates))]
        if all_bmi_cates[0]=="Underweight":
            bmi1.append(k)
        elif all_bmi_cates[0]=="Normal weight":
            bmi2.append(k)
        elif all_bmi_cates[0]=="Overweight":
            bmi3.append(k)
        elif all_bmi_cates[0]=="Obese":
            bmi456.append(k)
        elif all_bmi_cates[0]==-999:
            no_bmi_0.append(k)
        else:
            print(k) # should not happen
    no_bmi = list(set(df_bmi['subjectkey'])-set(bmi1)-set(bmi2)-set(bmi3)-set(bmi456))
    logger.info(
        "Baseline classification: %d underweight, %d normal, %d overweight, %d obese, %d bmi (missing bmi records)",
        len(bmi1), len(bmi2), len(bmi3), len(bmi456), len(no_bmi)
    )
    # Add BMI status labels to the main DataFrame
    bmi_status = [''] * len(df_bmi_1)
    for idx,bmi in enumerate(df_bmi_1['subjectkey']):
        if bmi in bmi1:
            bmi_status[idx] = 'bmi1'
        if bmi in bmi2:
            bmi_status[idx] = 'bmi2'
        if bmi in bmi3:
            bmi_status[idx] = 'bmi3'
        if bmi in bmi456:
            bmi_status[idx] = 'bmi456'
        if bmi in no_bmi:
            bmi_status[idx] = 'no_bmi'
    df_bmi_1['bmi_status'] = bmi_status
    
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
    df_madrs['is_madrs'] = [1 for _ in range(len(df_madrs))]
    df_ymrs['is_ymrs'] = [1 for _ in range(len(df_ymrs))]
    
    # Merge with MADRS and YMRS data
    df_bmi_2 = df_bmi_1.merge(
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
    logger.info("Labeling scheduled visits, based on the validness of MADRS/YMRS record...")
    is_scheduled = [0 for _ in range(len(df_bmi_4))]
    for idx, ism in enumerate(tqdm(df_bmi_4['is_madrs'])):
        if not (np.isnan(ism) and np.isnan(df_bmi_4['is_ymrs'][idx])):
            is_scheduled[idx] = 1
    df_bmi_4['is_scheduled'] = is_scheduled
    
    # Compute scheduled visits
    df_bmi_5 = compute_schedule_visit(df_bmi_4)

    # Map clinical status into 3 classes
    df_bmi_5['clinstat'] = df_bmi_5['clinstat'].astype('int')
    df_bmi_5['clinstat_1'] = df_bmi_5['clinstat'].apply(map_clinstat)
    
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
    logger.info("Event assignment complete: resulting rows = %d", df_new.shape[0])

    # Ensure only the first event occurrence is kept per subject
    logger.info("Filtering for first event occurrence per subject...")
    df_final = pd.DataFrame([], columns=df_new.columns)
    for idx,s in enumerate(tqdm(list(set(df_new['subjectkey'])))):
        df_temp = df_new[df_new['subjectkey']==s]
        first_one_index = df_temp.drop(df_temp[df_temp['event_occurs'].eq(1)].index[1:])
        df_final = pd.concat([df_final,first_one_index],axis=0)
    # Final event occurrence file in need to for plotting and Cox regression
    out_path = os.path.join(output_dir, 'stepBD_bmi_final_event_occurrence.csv')
    df_final.to_csv(out_path, index=False)
    logger.info("Final entries saved to %s (%d rows)", out_path, df_final.shape[0])
    return out_path, df_final


def process_stepbd_bmi_data(input_file, baseline_file, output_dir, madrs_file, ymrs_file):
    """
    Process and save BMI data for survival analysis.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the processed data.
    """
    # Process the data
    out_path, df_processed = preprocess_stepbd_bmi_data(input_file, baseline_file, output_dir, madrs_file, ymrs_file)
    
    # Save the processed data
    print(f"Processed smoking data saved to {out_path}")
    
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
    baseline_file = '../STEP_BD_pipelined/ade01.txt'
    output_dir = '../data4survivals_1/'
    madrs_file  = '../STEP-BD data/Text files from NDA/madrs01.txt'
    ymrs_file  = '../STEP-BD data/Text files from NDA/ymrs01.txt'
    
    # Process the data
    process_stepbd_bmi_data(input_file, baseline_file, output_dir, madrs_file, ymrs_file)

if __name__ == "__main__":
    main()
