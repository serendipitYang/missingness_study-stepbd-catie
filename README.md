# Longitudinal Study Data Analysis Pipeline

This is the analysis toolkit for evaluating missingness patterns and risk factors in longitudinal psychiatric studies, STEP-BD and CATIE, with focus on clinical severity, smoking status, and BMI as predictors of participant dropout.

This repository contains the data processing and analysis pipeline for the paper "Associations of missing visits with symptom severity, smoking, and BMI in longitudinal studies of severe mental illness".

## Overview

The pipeline processes and analyzes data from two major studies:
1. **CATIE** (Clinical Antipsychotic Trials of Intervention Effectiveness) - a randomized effectiveness trial in people with schizophrenia ([CATIE](https://www.nimh.nih.gov/funding/clinical-research/practical/catie#:~:text=The%20NIMH%2Dfunded%20Clinical%20Antipsychotic,medications%20used%20to%20treat%20schizophrenia.)) [[1]](#reference-1)
2. **STEP-BD** (Systematic Treatment Enhancement Program for Bipolar Disorder) - an observational study in people with bipolar disorder ([Step-BD](https://www.nimh.nih.gov/funding/clinical-research/practical/step-bd)) [[2]](#reference-2)

The analysis investigates whether missing visits in these studies are associated with clinical illness severity, smoking status, or BMI using survival analysis methods.

## Repository Structure

### Data Processing Scripts

- **`catie_bmi_processing.py`** - Processes CATIE BMI data for survival analysis
- **`catie_smoking_processing.py`** - Processes CATIE smoking data for survival analysis  
- **`stepbd_bmi_processing.py`** - Processes STEP-BD BMI data for survival analysis
- **`stepbd_smoking_processing.py`** - Processes STEP-BD smoking data for survival analysis

### Analysis Scripts

- **`survival_analysis.py`** - Main analysis script that generates Kaplan-Meier curves and Cox regression tables
- **`utils.py`** - Utility functions for data processing, visualization, and statistical analysis

## Key Features

### Data Processing Pipeline
- **Last Observation Carried Forward (LOCF) imputation** for missing clinical assessments
- **BMI categorization** (underweight, normal, overweight, obese) based on baseline measurements
- **Smoking status classification** from baseline visits
- **Clinical severity mapping** using PANSS (schizophrenia) and MADRS/YMRS (bipolar) scores
- **Event occurrence detection** for dropout patterns based on missing consecutive visits

### Statistical Analysis
- **Kaplan-Meier survival curves** stratified by risk factors and clinical severity
- **Cox proportional hazards regression** with pairwise comparisons
- **False Discovery Rate (FDR) correction** for multiple comparisons
- **Weighted analysis** to account for repeated measures per participant

### Visualization
- Publication-ready Kaplan-Meier plots comparing survival curves across groups
- Automated table generation for Cox regression results with proper formatting
- Excel output with statistical formatting for manuscript preparation

## Installation

```bash
# Clone the repository
git clone https://github.com/serendipitYang/missingness_study-stepbd-catie.git
cd missingness_study-stepbd-catie
```

# Install required packages
```bash
pip install pandas numpy matplotlib lifelines tqdm statsmodels xlsxwriter
```

## Usage

### 1. Data Preparation

Ensure your data files are organized as follows:

```
data/
├── CATIE/
│   ├── VISIT.csv
│   ├── vitals01.txt
│   ├── panss01.txt
│   ├── clgry01.txt
│   └── cgis01.txt
└── STEP-BD/
    ├── step_bd_final_data.csv
    ├── ade01.txt
    ├── madrs01.txt
    └── ymrs01.txt
```

### 2. Data Processing

Process each dataset individually:

# CATIE BMI processing
```python
from catie_bmi_processing import process_catie_bmi_data
process_catie_bmi_data(
    input_file='data/CATIE/VISIT.csv',
    output_dir='results/',
    vitals_file='data/CATIE/vitals01.txt',
    panss_file='data/CATIE/panss01.txt',
    clgry_file='data/CATIE/clgry01.txt'
)
```
# STEP-BD smoking processing  
```python
from stepbd_smoking_processing import process_smoking_data
process_smoking_data(
    input_file='data/STEP-BD/step_bd_final_data.csv',
    baseline_file='data/STEP-BD/ade01.txt',
    output_dir='results/',
    madrs_file='data/STEP-BD/madrs01.txt',
    ymrs_file='data/STEP-BD/ymrs01.txt'
)
```

### 3. Statistical Analysis

Run the complete analysis pipeline:

```python
from survival_analysis import main
main()
```

This generates:

- **Figure 2**: Kaplan-Meier curves for smoking status × clinical severity
- **Figure 3**: Kaplan-Meier curves for BMI categories × clinical severity
- **Table 2**: Cox regression results for smoking comparisons
- **Table 3**: Cox regression results for BMI comparisons

## Output Files

The analysis generates several output files:

### Processed Datasets

- `stepBD_smoking_final_event_occurrence.csv`
- `CATIE_smoking_final_event_occurrence.csv`
- `stepBD_bmi_final_event_occurrence.csv`
- `CATIE_bmi_final_event_occurrence.csv`

### Figures

- `Figure_2_smoking_clinstat_km_curves.png` - Smoking survival curves
- `Figure_3_bmi_clinstat_km_curves.png` - BMI survival curves

### Tables

- `Table_2_smoking_pairwise_comparisons.xlsx` - Smoking Cox regression results
- `Table_3_BMI_pairwise_comparisons.xlsx` - BMI Cox regression results

## Key Methods

### Clinical Severity Classification

- **CATIE**: Based on PANSS total scores (<58 = well, ≥58 = unwell) or Calgary Depression Scale (≤6 = well, >6 = unwell)
- **STEP-BD**: Based on MADRS and YMRS scores using established clinical thresholds

### BMI Categories

- Underweight: BMI < 18.5
- Normal weight: 18.5 ≤ BMI < 25
- Overweight: 25 ≤ BMI < 30
- Obese: BMI ≥ 30

### Dropout Definition

Participants are considered to have dropped out after missing 2 consecutive scheduled visits or reaching the end of follow-up.

### Statistical Comparisons

**Smoking Analysis** (Reference: Well non-smokers):

- People who smoke (PWS) vs. well people who don't smoke (PWDS)
- Unwell participants vs. well PWDS
- Unwell PWS vs. well PWDS

**BMI Analysis** (Reference: Well normal weight):

- Overweight vs. well normal weight
- Obese vs. well normal weight
- Unwell vs. well normal weight
- Unwell overweight vs. well normal weight
- Unwell obese vs. well normal weight

## Dependencies

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `lifelines` - Survival analysis
- `tqdm` - Progress bars
- `statsmodels` - Statistical modeling (FDR correction)
- `xlsxwriter` - Excel file generation

## Data Requirements

### CATIE Data Files

- `VISIT.csv` - Visit tracking and demographics
- `vitals01.txt` - Height, weight, and BMI measurements
- `panss01.txt` - PANSS psychiatric assessments
- `clgry01.txt` - Calgary Depression Scale scores
- `cgis01.txt` - Smoking and substance use data

### STEP-BD Data Files

- `step_bd_final_data.csv` - Main longitudinal dataset
- `ade01.txt` - Baseline demographics and characteristics
- `madrs01.txt` - Montgomery-Åsberg Depression Rating Scale
- `ymrs01.txt` - Young Mania Rating Scale


## License

This project is intended for academic and research use. Please cite the associated publication when using this code.

## References

<a id="reference-1"></a>
[1] Stroup TS, McEvoy JP, Swartz MS, et al. The National Institute of Mental Health Clinical Antipsychotic Trials of Intervention Effectiveness (CATIE) project: schizophrenia trial design and protocol development. Schizophr Bull. 2003;29: 15–31.

<a id="reference-2"></a>
[2] Sachs GS, Thase ME, Otto MW, et al. Rationale, design, and methods of the systematic treatment enhancement program for bipolar disorder (STEP-BD). Biol Psychiatry. 2003;53(11):1028-1042. doi:10.1016/s0006-3223(03)00165-3

## Contact

For questions about the analysis pipeline or data processing methods, please refer to the associated publication or contact the repository maintainers.
