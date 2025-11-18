# Anti-Money Laundering (AML) Detection with IBM Synthetic Dataset

**Author:** Meg Cuddihy

**Date**: November 2025

**Tools Used:** Python (Pandas, Scikit-learn, XGBoost, SHAP), Word

This repository contains a project that demonstrates an end-to-end workflow for detecting potentially suspicious transactions using machine learning and diagnostic reporting. 

It demonstrates skills in:

- Data exploration and analysis
    
- Feature engineering and selection
  
- Model building and hyperparameter tuning

- Performance evaluation and recommendations for future analysis
    
- Automated reporting using Python and Word templates

The model diagnostics focussed on minimising false negatives (i.e. missed suspicious transactions). This is based on the assumption that the cost of missing money-laundering activity outweighs the cost required to investigate transactions uneccessarily. 
  
The repository includes the full project structure and core analytical components. Additional enhancements are planned to progress this project towards a production-ready work pipeline to produce actionable insights to protect Australia's financial system from misuse.

---

## Repository Structure
```bash
aml_ibm_project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── templates/
│   ├── aml_template.docx
│   └── aoc_image.png
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.pyauto_report.py
│   ├── plot_utils.py
│   ├── model_utils.py
│   ├── report_utils.py
│   └── auto_report.py
├── outputs/
│   ├── report/
│   └── plots/
├── final_model.py
├── Makefile
├── data_dictionary.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

```


---

## Project Overview

The goal of this project is to explore, model and report on suspicious financial transactions using the **IBM AML transactions dataset** from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

The project simulates a workflow relevant to financial regulation analysts:

1. **Data exploration & cleaning:** data types, distributions, missing value checks, visualisations 
2. **Feature engineering:** deriving new features out of existing ones, behaviour features, rule-based flags for inclusion
3. **Machine Learning Modelling:** baseline XGBoost modelling, stratified train/test split, hyperparameter tuning 
4. **Evaluation of classification performance:** confusion matrix with emphasis on recall due to contextual importance of false negatives minimisation  
5. **Surface-level explainability via feature importance:** SHAP analysis, cluster analysis of false negatives and false positives  
6. **Communication of findings through automated report:** using pypandoc to read in variables and plots from analysis to provide insights to stakeholders using template

---

## Machine Learning Approach

The modelling focuses on predicting the binary classification:  
**Flagged vs. non-flagged transactions.**

Techniques used include:

- Feature Selection 
- Gradient Boosted Trees
- K-fold Cross-Validation
- Class imbalance handling 
- Train-test split with reproducibility  
- Precision, recall, F1 score  

Feature importance is extracted to support explainability.

---

## Automated Report Generation

The script `generate_report.py` fills a Word template (`aml_report_template.docx`) with a report summarising key insights from the data for stakeholder communication. 

Generated reports are saved in: `outputs/report/`.

---

## Next Steps

Extended hyperparameter tuning

Additional domain-specific featuring engineering to boost signals in behavioural differences in suspicious transactions

Further SHAP analysis and threshold optimisation

Improved documentation and cod comments

Completion and refinement of report content

Design and execution of an interactive Power BI dashboard to enable end-user exploration of features and model outputs

---

## Dataset Citation

This project uses the publicly available **IBM Transactions for Anti-Money Laundering (AML)** dataset.

**Recommended citation:**

Altman, E. (2019). *IBM Transactions for Anti-Money Laundering (AML)* [Data set]. Kaggle.  
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Please refer to the original dataset licence before reuse.

---

## Environment Setup

This project is compatible with both Conda and standard Python virtual environments. Choose the setup method that works best for your system:

### Option 1: Using Conda (recommended)

```bash
conda create -n aml python=3.10
conda activate aml
pip install -r requirements.txt
```
### Option 2: Using a Virtual Environment (no Conda required)

```bash
python -m venv .venv
# Activate the environment:
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## How to Run the Report Script

Once your environment is set up and dependencies are installed:

1. Train the model and generate metrics:
```bash
python src/final_model.py
```

2. Generate the Word report:
```bash
python src/auto_report.py
```

The following output will be saved to the `outputs/report/` folder:

- `aml_diagnostic_report.docx`: A formatted Word report with key findings and visualisations


## Data Limitations

This is a synthetic dataset, not real-world data. Data was created from IBM synthetic data generator.

Suspicious transaction patterns do not fully reflect the spectrum of Australian reporting behaviours.

Variables are simplified compared to real financial intelligence datasets.

No customer-level linkage or network analysis capability in the synthetic version.

Results should be interpreted as a demonstration of method, not of operational insights.

An accompanying Power BI dashboard is planned for a future project to allow end users to explore the data further. 

## Licence

This repository is released under the MIT Licence. This permits reuse for learning, review, and demonstration.

