# Anti-Money Laundering (AML) Detection with IBM Synthetic Dataset

**Author:** Meg Cuddihy

**Date**: November 2025

**Tools Used:** Python (Pandas, Scikit-learn, SHAP), Power BI, Word

This repository contains a project that showcases the application of analytical reasoning, coding and data visualisation in a financial intelligence context.

It demonstrates skills in:

- Python data analysis and machine learning
    
- Suspicious transaction detection
  
- Feature engineering and model evaluation
    
- Automated reporting using Python and Word templates
    
- Power BI data visualisation
  
- Clear analytical communication and data limitations awareness

---

## Repository Structure
```bash

├── data/
│ ├── raw/
│ └── processed/
├── notebooks/ 
├── powerbi/
│ └── aml_dashboard.pbix
├── report_template/
│ └── aml_report_template.docx 
├── src/
│ ├── model_training.py 
│ └── generate_report.py 
├── outputs/
│ ├── report/
│ └── figures/
├── .gitignore
├── LICENSE
└── README.md
```


---

## Project Overview

The goal of this project is to explore, model and report on suspicious financial transactions using the **IBM AML transactions dataset**.  
The project simulates a workflow relevant to financial regulation analysts:

1. **Data exploration & cleaning**  
2. **Feature engineering**  
3. **Machine Learning Modelling**  
4. **Evaluation of classification performance**  
5. **Surface-level explainability via feature importance**  
6. **Communication of findings through dashboards and reports**

---

## Machine Learning Approach

The modelling focuses on predicting the binary classification:  
**Flagged vs. non-flagged transactions.**

Techniques used include:

- Random Forest  
- Gradient Boosted Trees
- Support Vector Machines  
- Class imbalance handling 
- Train-test split with reproducibility  
- ROC-AUC, precision, recall, F1 score  

Feature importance is extracted to support explainability.

---

## Automated Report Generation

The script `generate_report.py` fills a Word template (`aml_report_template.docx`) with:

- Date of analysis  
- Model name  
- ROC-AUC, F1 score, precision, recall  
- Top predictive features  
- Key findings and limitations  

Generated reports are saved in: `outputs/report/`.


> These files are ignored by Git so that only the **template**, not personal results, is stored in the repository.

---

## Power BI Dashboard

A Power BI report is included to visualise:

- Transaction patterns  
- Suspicious activity indicators  
- Behaviour across customer segments  
- Model outputs (optional import of predictions)

The `.pbix` file is stored under `powerbi/`.

---

## Dataset Citation

This project uses the publicly available **IBM Transactions for Anti-Money Laundering (AML)** dataset.

**Recommended citation:**

Altman, E. (2019). *IBM Transactions for Anti-Money Laundering (AML)* [Data set]. Kaggle.  
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Please refer to the original dataset licence before reuse.

---

## How to Run the Report Script

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model and generate metrics:
```bash
python src/model_training.py
```

3. Generate the Word report:
```bash
python src/generate_report.py
```

The filled report will appear in:
```bash
outputs/report/
```

## Data Limitations

This is a synthetic dataset, not real-world data. Data was created from IBM synthetic data generator.

Suspicious transaction patterns do not fully reflect the spectrum of Australian reporting behaviours.

Variables are simplified compared to real financial intelligence datasets.

No customer-level linkage or network analysis capability in the synthetic version.

Results should be interpreted as a demonstration of method, not of operational insights.

## Licence

This repository is released under the MIT Licence. This permits reuse for learning, review, and demonstration.

