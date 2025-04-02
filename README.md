This is the analysis toolkit for evaluating missingness patterns and risk factors in longitudinal psychiatric studies, STEP-BD and CATIE , with focus on clinical severity, smoking status, and BMI as predictors of participant dropout.

# Longitudinal Study Data Analysis Pipeline
This repository contains the data processing and analysis pipeline for the paper "Associations of missing visits with symptom severity, smoking, and BMI in longitudinal studies of severe mental illness".

## Overview

The pipeline processes and analyzes data from two major studies:
1. **CATIE** (Clinical Antipsychotic Trials of Intervention Effectiveness) - a randomized effectiveness trial in people with schizophrenia ([CATIE](https://www.nimh.nih.gov/funding/clinical-research/practical/catie#:~:text=The%20NIMH%2Dfunded%20Clinical%20Antipsychotic,medications%20used%20to%20treat%20schizophrenia.)) [[1]](#reference-1);
2. **STEP-BD** (Systematic Treatment Enhancement Program for Bipolar Disorder) - an observational study in people with bipolar disorder ([Step-BD](https://www.nimh.nih.gov/funding/clinical-research/practical/step-bd))[[2]](#reference-2).

The analysis investigates whether missing visits in these studies are associated with clinical illness severity, smoking status, or BMI.

## Repository Structure

### Data Processing Scripts

- **CATIE Processing**
  - `catie_processing_utils.py`: Utility functions for CATIE data processing
  - `catie_smoking_processing.py`: Processes smoking-related data from CATIE
  - `catie_bmi_processing.py`: Processes BMI-related data from CATIE

- **STEP-BD Processing**
  - `stepbd_processing_utils.py`: Utility functions for STEP-BD data processing
  - `stepbd_smoking_processing.py`: Processes smoking-related data from STEP-BD
  - `stepbd_bmi_processing.py`: Processes BMI-related data from STEP-BD

### Analysis Scripts

- **Survival Analysis and Visualization**
  - `visualization_utils.py`: Functions for Kaplan-Meier curve generation
  - `cox_regression_utils.py`: Utilities for Cox proportional hazards modeling and statistical analysis
  - `survival_analysis.py`: Main script orchestrating the comprehensive analysis workflow

## Data Processing Pipeline

### 1. Data Loading and Preparation
- Load raw study data files
- Merge multiple datasets (e.g., visits, vitals, clinical assessments)
- Clean and standardize formats

### 2. Clinical Status Determination
- Calculate clinical status based on study-specific measures
  - PANSS and Calgary scores for CATIE
  - MADRS and YMRS for STEP-BD
- Apply LOCF (Last Observation Carried Forward) imputation for missing values

### 3. Participant Categorization
- Classify participants by smoking status (smoker, non-smoker, changer)
- Categorize participants by BMI (normal weight, overweight, obese)
- Track status changes across visits

### 4. Survival Analysis Preparation
- Calculate visit attendance and identify dropout events
- Create event indicators for survival analysis
- Compute proper weights for repeated measures

## Analysis Workflow

### 1. Visualization
- Generate Kaplan-Meier curves for:
  - Smoking status (smokers vs. non-smokers)
  - BMI categories (normal weight, overweight, obese)
  - Stratification by clinical status (well vs. unwell)

### 2. Statistical Modeling
- Run Cox proportional hazards models:
  - Basic models with main effects and interactions
  - Pairwise comparisons between specific groups
  - Calculate hazard ratios with confidence intervals

### 3. Multiple Testing Corrections
- Apply Benjamini-Hochberg procedure for p-value adjustment
- Generate final statistics for publication

## Usage

### Requirements
- Python 3.8+
- pandas
- numpy
- lifelines
- matplotlib
- statsmodels
- tqdm

### Running the Pipeline

1. **Data Processing**
   ```bash
   # Process CATIE data
   python catie_smoking_processing.py
   python catie_bmi_processing.py
   
   # Process STEP-BD data
   python stepbd_smoking_processing.py
   python stepbd_bmi_processing.py
   ```

2. **Analysis**
   ```bash
   # Run full analysis pipeline
   python survival_analysis.py
   ```

## Customization

- File paths can be modified in the `main()` function of each script
- Analysis parameters can be adjusted in the respective utility files
- Visualization options can be customized in `visualization_utils.py`

## Key Function Reference

### Data Processing
- `locf_imputation()`: Impute missing values using Last Observation Carried Forward
- `map_clinstat()`: Map clinical measures to clinical status categories
- `find_missing_required_visits()`: Identify dropout events based on missing visits

### Analysis
- `prepare_data_for_visualization()`: Prepare datasets for KM visualization
- `plot_km_smoking_curves()`: Generate Kaplan-Meier curves for smoking status
- `plot_km_smoking_clinstat_curves()`: Generate KM curves for smoking status stratified by clinical status
- `plot_km_bmi_curves()`: Generate Kaplan-Meier curves for BMI categories
- `plot_km_bmi_clinstat_curves()`: Generate KM curves for BMI categories stratified by clinical status
- `run_smoking_pairwise_comparisons()`: Run Cox models comparing smoking-related groups
- `run_bmi_pairwise_comparisons()`: Run Cox models comparing BMI-related groups
- `adjust_pvalues()`: Apply multiple testing corrections

## Output

- Processed CSV files with participant data ready for survival analysis
- Kaplan-Meier curves saved as PNG files
- Statistical model results with adjusted p-values

## References

<a id="reference-1"></a>
[1] Stroup TS, McEvoy JP, Swartz MS, et al. The National Institute of Mental Health Clinical Antipsychotic Trials of Intervention Effectiveness (CATIE) project: schizophrenia trial design and protocol development. Schizophr Bull. 2003;29: 15–31.

<a id="reference-2"></a>
[2] Sachs GS, Thase ME, Otto MW, et al. Rationale, design, and methods of the systematic treatment enhancement program for bipolar disorder (STEP-BD). Biol Psychiatry. 2003;53(11):1028-1042. doi:10.1016/s0006-3223(03)00165-3
