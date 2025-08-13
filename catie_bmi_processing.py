import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import warnings

from utils import (
    locf_imputation, 
    panss_cols,
    clgry_cols,
    map_clinstat_panss_or_calg,
    get_visit_id_catie,
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

def calculate_bmi(height_inches, weight_lbs):
    """
    Calculate BMI (Body Mass Index) using height in inches and weight in pounds.
    
    Args:
        height_inches (float): Height in inches
        weight_lbs (float): Weight in pounds
        
    Returns:
        float: BMI value rounded to one decimal place
    """
    if np.isnan(height_inches) or np.isnan(weight_lbs):
        return np.nan
    
    # Convert height from inches to meters
    height_meters = height_inches * 0.0254
    
    # Convert weight from pounds to kilograms
    weight_kg = weight_lbs * 0.453592
    
    # Calculate BMI using the formula: weight (kg) / height^2 (m^2)
    bmi = weight_kg / (height_meters ** 2)
    
    # Round to one decimal place
    return round(bmi, 1)

def bmi_mapping(bmi_value):
    """
    Map BMI values to categories.
    
    Args:
    bmi_value (float): BMI value
    
    Returns:
    int: BMI category (1: underweight, 2: normal, 3: overweight, 4: obese)
    """
    if pd.isna(bmi_value):
        return np.nan
    
    try:
        bmi_value = float(bmi_value)
    except:
        return np.nan
    
    if bmi_value < 18.5:
        return 1
    elif bmi_value < 25: 
        return 2
    elif bmi_value < 30:
        return 3
    else: # >=30
        return 4

def preprocess_catie_bmi_data(input_file, output_dir, vitals_file, panss_file, clgry_file):
    """
    Preprocess CATIE BMI data.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Directory to save processed data.
    
    Returns:
    DataFrame: Processed BMI data.
    """
    # Read visit data
    logger.info("Reading visit data...")
    df_all = pd.read_csv(input_file, skiprows=1, names=[
        'CATIEID', 'subjectkey', 'gender', 'visit', 'visitid', 'TRUNCVIS',
        'PHASE', 'MISSED', 'ATTENDED', 'DISCON', 'VISDAY'
    ])
    set_all = set(df_all['subjectkey'])
    # Read essential datasets, with selected columns
    df_vitals = pd.read_csv(vitals_file, delimiter="\t")
    df_vitals = df_vitals.iloc[1:,:] # Skip the first row which is a sub-header
    df_vitals = df_vitals[['subjectkey','visit', 'visitid','bmi', 'height_std', 'weight_std']].sort_values(['subjectkey', 'visit'])

    df_panss = pd.read_csv(panss_file, skiprows=2, delimiter="\t", names=panss_cols)
    df_clgry = pd.read_csv(clgry_file, skiprows=2, delimiter="\t", names=clgry_cols)
    
    # Prepare the rest data for merging
    convert_columns = ['visit','visitid', 'subjectkey']
    for df in [df_all, df_vitals, df_panss, df_clgry]:
        for col in convert_columns:
            df[col] = df[col].astype('str')

    # Merge datasets
    logger.info("Merging datasets...")
    df_bmi = df_all.merge(df_vitals[['subjectkey', 'visit', 'visitid', 'height_std', 'weight_std']], 
                           on=['subjectkey', 'visit', 'visitid'], how='left')
    df_bmi = df_bmi.merge(df_panss[['subjectkey', 'visit', 'visitid', 'panss_general',
                                     'panss_total', 'panss_positive', 'panss_negative']], 
                           on=['subjectkey', 'visit', 'visitid'], how='left')
    df_bmi = df_bmi.merge(df_clgry[['subjectkey', 'visit', 'visitid', 'calg_ts']], 
                           on=['subjectkey', 'visit', 'visitid'], how='left')

    # Rename columns
    df_bmi.rename(columns={'VISDAY': 'daysrz'}, inplace=True)

    # Clinical status determination
    logger.info("Determining clinical status by PANSS or Calgary score...")
    df_bmi['clinstat_pt'] = map_clinstat_panss_or_calg(df_bmi, is_panss=True)
    df_bmi['clinstat_cg'] = map_clinstat_panss_or_calg(df_bmi, is_panss=False)
    df_bmi['clinstat'] = map_clinstat_panss_or_calg(df_bmi, is_panss=None)

    # Impute clinical status
    logger.info("Imputing clinical status...")
    columns_to_impute = ['clinstat']
    df_bmi_imputed = locf_imputation(df_bmi, columns_to_impute)
    df_bmi_imputed.dropna(subset=['clinstat'], inplace=True)
    df_bmi_imputed['clinstat_1'] = df_bmi_imputed['clinstat'].astype('int')
    set_clinstat_imputed = set(df_bmi_imputed['subjectkey'])
    logger.info(
        "%d subjects missed clinical status after imputation, out of %d total subjects",
        len(set_all) - len(set_clinstat_imputed), 
        len(set_all)
    )

    # Impute height and weight for BMI calculation
    # Calculate BMI for each row, based on height (only valid at baseline) and weight (updated/recorded all the time)
    logger.info("Imputing height and weight, and doing BMI calculation...")
    columns_to_impute = ['height_std', 'weight_std']
    df_bmi_2_imputed = locf_imputation(df_bmi_imputed, columns_to_impute)
    df_bmi_3 = df_bmi_2_imputed.dropna(subset=columns_to_impute)
    set_hieght_weight_imputed = set(df_bmi_3['subjectkey'])
    logger.info(
        "%d subjects missed height/weight after imputation, out of %d subjects since last imputation",
        len(set_clinstat_imputed) - len(set_hieght_weight_imputed), 
        len(set_clinstat_imputed)
    )
    no_bmi = list(set_clinstat_imputed - set_hieght_weight_imputed)
    # Create an empty DataFrame with the expected columns
    df_bmi_4 = pd.DataFrame([], columns=list(df_bmi_3.columns)+['bmi_calculated', 'bmi_cat_calculated'])

    df_bmi_3['height_std'] = df_bmi_3['height_std'].astype('float')
    df_bmi_3['weight_std'] = df_bmi_3['weight_std'].astype('float')
    for subject in tqdm(df_bmi_3['subjectkey'].unique()):
        subject_data = df_bmi_3[df_bmi_3['subjectkey'] == subject]
        
        # Fix this line - use a consistent height value for the subject
        subject_data['height_std'] = [list(subject_data['height_std'])[0] for _ in range(len(subject_data))]
        
        # Apply the BMI calculation row by row using apply
        subject_data['bmi_calculated'] = subject_data.apply(
            lambda row: calculate_bmi(row['height_std'], row['weight_std']), axis=1)
        subject_data['bmi_cat_calculated'] = subject_data['bmi_calculated'].apply(bmi_mapping)
        # Concatenate to the result DataFrame
        df_bmi_4 = pd.concat([df_bmi_4, subject_data])

    # Categorize BMI groups
    logger.info("Determining BMI status by Screening or Baseline visit...")
    bmi1, bmi2, bmi3, bmi456 = [], [], [], []
    for k in tqdm(list(set(df_bmi_4['subjectkey']))):
        bmi_cates = list(df_bmi_4[df_bmi_4['subjectkey']==k]['bmi_cat_calculated'])
        all_bmi_cates = [int(s) for s in list(set(bmi_cates))]
        if all_bmi_cates[0]==1:
            bmi1.append(k)
        elif all_bmi_cates[0]==2:
            bmi2.append(k)
        elif all_bmi_cates[0]==3:
            bmi3.append(k)
        elif all_bmi_cates[0]>=4:
            bmi456.append(k)
        else:
            print(k) # should not happen
    logger.info(
        "Baseline classification: %d underweight, %d normal, %d overweight, %d obese, %d no_bmi (missing bmi records)",
        len(bmi1), len(bmi2), len(bmi3), len(bmi456), len(no_bmi)
    )

    # Mark BMI status
    bmi_status = ['' for _ in range(len(df_bmi_4))]
    for idx, bmi in enumerate(df_bmi_4['subjectkey']):
        if bmi in bmi1:
            bmi_status[idx] = 'bmi1'
        if bmi in bmi2:
            bmi_status[idx] = 'bmi2'
        if bmi in bmi3:
            bmi_status[idx] = 'bmi3'
        if bmi in bmi456:
            bmi_status[idx] = 'bmi456'
    df_bmi_4['bmi_status'] = bmi_status

    # Prepare for survival analysis
    df_bmi_5 = df_bmi_4[
        df_bmi_4['visit'].str.match(r'^(Visit\d+|Baseline|Screening)$', na=False)
    ]

    # Prepare final dataset for survival analysis
    logger.info("Computing event occurrences for %d subjects...", len(set(df_bmi_5['subjectkey'])))
    df_new = pd.DataFrame([], columns=list(df_bmi_5.columns)+['visit_id', 'event_occurs'])
    for idx, subject in enumerate(tqdm(list(set(df_bmi_5['subjectkey'])))):
        subject_df = df_bmi_5[df_bmi_5['subjectkey'] == subject]
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

    out_path = os.path.join(output_dir, 'CATIE_bmi_final_event_occurrence.csv')
    df_final.to_csv(out_path, index=False)
    logger.info("Final entries saved to %s (%d rows)", out_path, df_final.shape[0])
    return out_path, df_final

def process_catie_bmi_data(input_file, output_dir, vitals_file, panss_file, clgry_file):
    """
    Process and save CATIE BMI data for survival analysis.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the processed data.
    missing_visits (int): Number of missing visits to define dropout.
    """
    # Process the data
    out_path, df_processed  = preprocess_catie_bmi_data(input_file, output_dir, vitals_file, panss_file, clgry_file)
    
    # Save the processed data
    print(f"Processed CATIE BMI data saved to {out_path}")
    
    # Print basic statistics
    print("\nData Processing Statistics:")
    print("Total subjects:", len(set(df_processed['subjectkey'])))
    print("BMI status distribution:")
    print(Counter(df_processed['bmi_status']))
    print("\nEvent occurrence distribution:")
    print(Counter(df_processed['event_occurs']))

def main():
    """
    Main function to run the CATIE BMI data processing.
    """
    # Paths - replace these with your actual paths
    input_file = '../CATIE/CATIE data from NIH NDA/VISIT.csv'
    vitals_file = '../CATIE/CATIE data from NIH NDA/vitals01.txt'
    panss_file = '../CATIE/CATIE data from NIH NDA/panss01.txt'
    clgry_file = '../CATIE/CATIE data from NIH NDA/clgry01.txt'
    output_dir = '../data4survivals_1/'
    
    # Process the data
    process_catie_bmi_data(input_file, output_dir, vitals_file, panss_file, clgry_file)

if __name__ == "__main__":
    main()
