# Supplementary Materials Analysis

## Overview

The `supplement.py` module generates comprehensive supplementary analyses for the smoking and BMI survival studies. This includes population overlap analyses, main effect analyses, and multivariate Cox regression models.

## Generated Outputs

### Figures

1. **Supplement Figure 1**: Population Overlap Venn Diagrams
   - Two Venn diagrams (StepBD and CATIE)
   - Shows overlap between Smoking Study and BMI Study populations
   - Three sets: "Only in Smoking Study", "Only in BMI Study", "In Both Studies"

2. **Supplement Figure 2**: KM Curves for Exclusive Populations
   - Compares survival curves for subjects only in Smoking vs. only in BMI studies
   - Separate plots for StepBD and CATIE
   - StepBD: 983 subjects (10,466 obs) in Smoking only, 152 subjects (1,543 obs) in BMI only
   - CATIE: 37 subjects (499 obs) in Smoking only, 4 subjects (45 obs) in BMI only
   - Note: Exclusions (missing baseline, underweight) applied before creating exclusive populations

3. **Supplement Figure 3**: Smoking Main Effect
   - KM curves comparing PWS (smokers) vs. PWDS (non-smokers) on whole population
   - Not stratified by clinical status (univariate analysis)

4. **Supplement Figure 4**: BMI Main Effect
   - KM curves for Normal Weight, Overweight, and Obese groups
   - Univariate analysis across all clinical statuses

5. **Supplement Figure 5**: Illness Severity Main Effect
   - KM curves comparing Well vs. Unwell patients
   - Univariate analysis

### Tables

1. **Supplement Table 1**: Population Overlap Statistics
   - CSV format
   - Counts for exclusive and overlapping populations

2. **Supplement Table 2**: Smoking Main Effect Cox Regression
   - CSV and Excel formats
   - Columns: Dataset, Comparison, Number of Observations, HR, HR lower/upper 95%, z, Uncorrected p, FDR-corrected p
   - PWS vs. PWDS comparison

3. **Supplement Table 3**: BMI Main Effect Cox Regression
   - CSV and Excel formats
   - Same column structure
   - Comparisons: Overweight vs. Normal, Obese vs. Normal

4. **Supplement Table 4**: Illness Severity Main Effect Cox Regression
   - CSV and Excel formats
   - Same column structure
   - Unwell vs. Well comparison

5. **Supplement Table 5**: Multivariate Cox - Smoking + Illness Severity
   - CSV and Excel formats
   - Same comparisons as Table 2, but with multivariate adjustment
   - 3 rows per dataset (6 total):
     1. PWS vs. well PWDS (adjusted for illness severity)
     2. Unwell vs. well PWDS (adjusted for smoking status)
     3. Unwell PWS vs. well PWDS (unadjusted - shows combined effect)
   - Adjusted models control for confounding by the other variable

6. **Supplement Table 6**: Multivariate Cox - BMI + Illness Severity
   - CSV and Excel formats
   - Same comparisons as Table 3, but with multivariate adjustment
   - 5 rows per dataset (10 total):
     1. Overweight vs. well normal weight (adjusted for illness)
     2. Obese vs. well normal weight (adjusted for illness)
     3. Unwell vs. well normal weight (adjusted for BMI)
     4. Unwell overweight vs. well normal weight (unadjusted - combined effect)
     5. Unwell obese vs. well normal weight (unadjusted - combined effect)
   - Adjusted models control for confounding

## Usage

### Basic Usage

```python
from supplement import run_supplementary_analyses

# Define input file paths
stepbd_smoking_path = '../data4survivals_1/stepBD_smoking_final_event_occurrence.csv'
catie_smoking_path = '../data4survivals_1/CATIE_smoking_final_event_occurrence.csv'
stepbd_bmi_path = '../data4survivals_1/stepBD_bmi_final_event_occurrence.csv'
catie_bmi_path = '../data4survivals_1/CATIE_bmi_final_event_occurrence.csv'

# Run all analyses
run_supplementary_analyses(
    stepbd_smoking_path, catie_smoking_path,
    stepbd_bmi_path, catie_bmi_path,
    output_dir='../data4survivals_1/supplement_results'
)
```

### Run from Command Line

```bash
cd /path/to/togithub
python supplement.py
```

### Individual Analyses

You can also run specific analyses:

```python
from supplement import *
import pandas as pd

# Load and prepare data
df_stepbd_smoking = pd.read_csv(stepbd_smoking_path)
df_catie_smoking = pd.read_csv(catie_smoking_path)
# ... prepare data using prepare_data_for_visualization()

# Run specific analysis
overlap_summary = create_population_overlap_report(
    df_stepbd_smoking, df_catie_smoking,
    df_stepbd_bmi, df_catie_bmi,
    output_dir='results'
)

smoking_main = plot_smoking_main_effect(
    df_stepbd_smoking, df_catie_smoking,
    output_dir='results'
)

multivar_results = create_multivariate_smoking_illness_table(
    df_stepbd_smoking, df_catie_smoking,
    output_dir='results'
)
```

## Key Differences from Main Analysis

### Main Analysis (survival_analysis.py)
- **Pairwise comparisons**: Each group compared to a specific reference group
- **Stratified by clinical status**: All analyses consider well vs. unwell separately
- **Focus**: Interaction effects between exposures and clinical status

### Supplement Analysis (supplement.py)
- **Main effects**: Overall effect of each variable across all groups
- **Multivariate models**: Adjusted effects controlling for confounders
- **Population overlap**: Understanding the relationship between study cohorts
- **Univariate analyses**: Simpler models showing overall trends

## Dependencies

Make sure to install all required packages:

```bash
pip install -r requirements.txt
```

New dependencies added for supplement analyses:
- `matplotlib-venn>=0.11.6` - For Venn diagram visualization
- `xlsxwriter>=3.0.0` - For Excel file writing

## Output Directory Structure

```
supplement_results/
├── Supplement_Figure_1_Population_Overlap.png
├── Supplement_Figure_2_KM_Exclusive_Populations.png
├── Supplement_Figure_3_Smoking_Main_Effect.png
├── Supplement_Figure_4_BMI_Main_Effect.png
├── Supplement_Figure_5_Illness_Severity_Main_Effect.png
├── Supplement_Table_1_Population_Overlap.csv
├── Supplement_Table_2_Smoking_Main_Effect.csv
├── Supplement_Table_2_Smoking_Main_Effect.xlsx
├── Supplement_Table_3_BMI_Main_Effect.csv
├── Supplement_Table_3_BMI_Main_Effect.xlsx
├── Supplement_Table_4_Illness_Severity_Main_Effect.csv
├── Supplement_Table_4_Illness_Severity_Main_Effect.xlsx
├── Supplement_Table_5_Multivariate_Smoking_Illness.csv
├── Supplement_Table_5_Multivariate_Smoking_Illness.xlsx
├── Supplement_Table_6_Multivariate_BMI_Illness.csv
└── Supplement_Table_6_Multivariate_BMI_Illness.xlsx
```

## Notes

1. **Weighting**: All analyses use the same weighting scheme as main analyses (inverse of number of visits per subject)

2. **FDR Correction**: P-values are corrected for multiple comparisons using the Benjamini-Hochberg method

3. **Excel Formatting**: Tables are automatically formatted with:
   - Counts with thousand separators
   - HR/CI values to 2 decimal places
   - P-values with <0.005 notation for very small values

4. **Exclusions Applied**:

   **Population Overlap Analysis:**
   - Smoking study: Excludes subjects with `smoker_yn_1 == 'missing_baseliner'`
     - StepBD: 3,835 subjects (removed 272 missing baseline)
     - CATIE: 1,447 subjects (removed 5 missing baseline)
   - BMI study: Excludes subjects with `bmi_status` in ['no_bmi', 'bmi1'] (missing baseline and underweight)
     - StepBD: 3,004 subjects (removed 571 underweight + 1,103 no BMI)
     - CATIE: 1,414 subjects (removed 368 underweight)

   **Main Effect Analyses:**
   - Smoking: Exclude "changers" (only compare consistent smokers vs. non-smokers)
   - BMI: Exclude underweight subjects (focus on normal, overweight, obese)
   - All: Exclude subjects with missing survival data (daysrz, event_occurs)

## Interpretation Guide

### Main Effect Tables
- **HR > 1**: Increased risk of dropout for the comparison group
- **HR < 1**: Decreased risk of dropout (protective effect)
- **FDR-corrected p < 0.05**: Statistically significant after multiple comparison correction

### Multivariate Tables (Tables 5 & 6)
- Shows **adjusted** hazard ratios for the same comparisons as main Tables 2 & 3
- Table 5 (Smoking + Illness):
  - PWS vs. well PWDS: Adjusted for illness severity (`comparison + clinstat`)
  - Unwell vs. well PWDS: Adjusted for smoking status (`comparison + smoker`)
  - Unwell PWS vs. well PWDS: Unadjusted (`comparison` only)
    - Shows total combined effect of both risk factors
- Table 6 (BMI + Illness):
  - BMI comparisons (overweight/obese vs. normal): Adjusted for illness (`comparison + clinstat`)
  - Illness comparison (unwell vs. well): Adjusted for BMI (`comparison + overweight + obese`)
  - Combined groups (unwell overweight/obese): Unadjusted (`comparison` only)
    - Show total combined effects
- **Adjusted models**: Control for confounding by other risk factors
- **Unadjusted models**: For combined groups, show total effect (confounding control not possible due to group definition)
- Useful for understanding independent vs. combined effects of risk factors

## Contact

For questions or issues, please refer to the main README.md or contact the study team.
