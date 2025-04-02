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
    else:
        return 4

def preprocess_catie_bmi_data(input_file, output_dir):
    """
    Preprocess CATIE BMI data.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Directory to save processed data.
    
    Returns:
    DataFrame: Processed BMI data.
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

    # Prepare BMI data
    df_vitals = df_vitals.iloc[1:].reset_index(drop=True)
    df_vitals['visitid'] = df_vitals['visitid'].astype('str')
    
    # Calculate BMI for each row
    df_vitals['bmi'] = df_vitals.apply(lambda row: calculate_bmi(row['height_std'], row['weight_std']), axis=1)
    df_vitals['bmi_cat'] = df_vitals['bmi'].apply(bmi_mapping)

    # Prepare data for merging
    convert_columns = ['visitid', 'subjectkey']
    for df in [df_all, df_vitals, df_panss, df_clgry]:
        for col in convert_columns:
            df[col] = df[col].astype('str')

    # Merge datasets
    df_bmi = df_all.merge(df_vitals[['subjectkey', 'visit', 'visitid', 'bmi', 'bmi_cat', 'height_std', 'weight_std']], 
                           on=['subjectkey', 'visit', 'visitid'], how='left')
    df_bmi = df_bmi.merge(df_panss[['subjectkey', 'visit', 'visitid', 
                                     'panss_total', 'panss_positive', 'panss_negative']], 
                           on=['subjectkey', 'visit', 'visitid'], how='left')
    df_bmi = df_bmi.merge(df_clgry[['subjectkey', 'visit', 'visitid', 'calg_ts']], 
                           on=['subjectkey', 'visit', 'visitid'], how='left')

    # Rename columns
    df_bmi.rename(columns={'VISDAY': 'daysrz'}, inplace=True)

    # Clinical status determination
    df_bmi['clinstat_pt'] = map_clinstat(df_bmi, is_panss=True)
    df_bmi['clinstat_cg'] = map_clinstat(df_bmi, is_panss=False)
    df_bmi['clinstat'] = map_clinstat(df_bmi, is_panss=None)

    # Impute clinical status
    columns_to_impute = ['clinstat']
    df_bmi_imputed = locf_imputation(df_bmi, columns_to_impute)
    df_bmi_imputed.dropna(subset=['clinstat'], inplace=True)
    df_bmi_imputed['clinstat_1'] = df_bmi_imputed['clinstat'].astype('int')

    # Categorize BMI groups
    bmi1, bmi2, bmi3, bmi456, transformer = [], [], [], [], []
    for k in tqdm(list(set(df_bmi_imputed['subjectkey']))):
        bmi_cates = list(df_bmi_imputed[df_bmi_imputed['subjectkey']==k]['bmi_cat'])
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

    # Mark BMI status
    bmi_status = [''] * len(df_bmi_imputed)
    for idx, bmi in enumerate(df_bmi_imputed['subjectkey']):
        if bmi in bmi1:
            bmi_status[idx] = 'bmi1'
        if bmi in bmi2:
            bmi_status[idx] = 'bmi2'
        if bmi in bmi3:
            bmi_status[idx] = 'bmi3'
        if bmi in bmi456:
            bmi_status[idx] = 'bmi456'
        if bmi in transformer:
            bmi_status[idx] = 'transformer'
    df_bmi_imputed['bmi_status'] = bmi_status

    # Prepare for survival analysis
    df_bmi_filtered = df_bmi_imputed[
        df_bmi_imputed['visit'].str.match(r'^(Visit\d+|Baseline|Screening)$', na=False)
    ]

    # Prepare final dataset for survival analysis
    df_new = pd.DataFrame([], columns=list(df_bmi_filtered.columns)+['visit_id', 'event_occurs'])
    for idx, subject in enumerate(tqdm(list(set(df_bmi_filtered['subjectkey'])))):
        subject_df = df_bmi_filtered[df_bmi_filtered['subjectkey'] == subject]
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

def process_catie_bmi_data(input_file, output_file, missing_visits=2):
    """
    Process and save CATIE BMI data for survival analysis.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the processed data.
    missing_visits (int): Number of missing visits to define dropout.
    """
    # Process the data
    df_processed = preprocess_catie_bmi_data(input_file, output_dir=os.path.dirname(output_file))
    
    # Save the processed data
    df_processed.to_csv(output_file, index=False)
    print(f"Processed CATIE BMI data saved to {output_file}")
    
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
    output_file = '../data4survivals/CATIE_2missing_bmi_ptORcg_imputed.csv'
    
    # Process the data
    process_catie_bmi_data(input_file, output_file)

if __name__ == "__main__":
    main()
