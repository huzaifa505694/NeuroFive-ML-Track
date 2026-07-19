# NeuroFive-ML-Track

Task submissions for the **Neurofive Solutions ML Internship Track** — a series of hands-on tasks building ML fundamentals from the ground up, documented publicly on GitHub and LinkedIn.

---

## Week 1 · Task 1 — Environment Setup & First Exploratory Data Analysis (EDA)

### Objective
Before touching any modeling, set up a working Python/ML toolkit and practice "listening" to a dataset — understanding its shape, quality, and quirks prior to building anything on top of it.

### Environment
| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Google Colab | Notebook environment (no local install needed) |
| pandas | Data loading and inspection |
| NumPy | Numerical operations, dtype filtering |

### Dataset
- **Source:** Kaggle — [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)
- **File used:** `train.csv` (891 rows, 12 columns)
- `test.csv` and `gender_submission.csv` were not used for this task.

### Steps Performed
1. Loaded the dataset with `pandas.read_csv()`
2. Inspected structure with `.info()`, `.describe()`, and `.head()`
3. Quantified missing values per column (count + percentage)
4. Split columns into **numerical** vs **categorical** by dtype
5. Summarized findings in a markdown "data story" cell

### Key Findings
- **Shape:** 891 rows × 12 columns
- **Missing data:**
  - `Cabin` — ~77% missing
  - `Age` — ~20% missing
  - `Embarked` — 2 rows missing
- **Numerical columns (7):** `PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare`
- **Categorical columns (5):** `Name, Sex, Ticket, Cabin, Embarked`
- **Target balance:** `Survived` is imbalanced — roughly 62% did not survive vs. 38% survived
- `Fare` is right-skewed with notable outliers, worth watching before any scaling/modeling step
- `Cabin`'s missing rate is high enough that it's better engineered into a binary "has_cabin" flag than imputed directly

### How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/neurofive-ml-track.git
   ```
2. Open `titanic_eda.ipynb` in [Google Colab](https://colab.research.google.com) or a local Jupyter Notebook.
3. Download `train.csv` from the [Kaggle Titanic data page](https://www.kaggle.com/competitions/titanic/data) and upload it to the same Colab session (or place it next to the notebook locally).
4. Run all cells top to bottom.

### Repository Structure
```
neurofive-ml-track/
├── titanic_eda.ipynb   # EDA notebook for Task 1
├── README.md           # This file
└── train.csv           # Not committed — download from Kaggle (see above)
```

### Deliverables Checklist
- [x] Python/Jupyter/pandas/NumPy environment working
- [x] Dataset loaded and inspected (`.info()`, `.describe()`, `.head()`)
- [x] Missing values and column types identified
- [x] Markdown "data story" written
- [x] Notebook pushed to this public repo
- [x] 2–3 min walkthrough recorded and posted to LinkedIn, tagging Neurofive Solutions

---

## Week 1 · Task 2 — Data Cleaning & Visual Storytelling

### Objective
Real data is messy. This task cleans the dataset properly and tells its story visually — treating visualization as a diagnostic tool for catching mistakes and finding patterns, not decoration.

### Missing Value Strategy
| Column | Missing | Method | Justification |
|---|---|---|---|
| `Age` | ~20% | Median fill | Age is right-skewed; median resists distortion from outliers better than the mean |
| `Embarked` | 2 rows | Mode fill | Only 2 rows affected — using the most common port is low-risk |
| `Cabin` | ~77% | Converted to binary `Has_Cabin` flag, column dropped | Too sparse to impute meaningfully; a presence/absence flag preserves signal without fabricating values |

### Outlier Detection
- **Method:** Boxplot on `Fare`
- **Finding:** Several extreme high-fare outliers well above the interquartile range, consistent with a small number of premium tickets

### Visualizations
1. **Histogram** — Age distribution
2. **Boxplot** — Fare (outlier detection)
3. **Bar chart** — Survival rate by Sex
4. **Correlation heatmap** — numerical feature correlations, including engineered `Has_Cabin`

### Key Question: Which feature most affects survival?
**Sex** shows the largest, most consistent gap in survival rate (~74% for women vs. ~19% for men), backed by `Pclass` and `Fare` also correlating with `Survived` in the heatmap — consistent with the historical "women and children first" account and first-class passengers having better lifeboat access.

### Deliverables Checklist
- [x] Missing values handled and justified in a markdown note
- [x] Outliers detected via boxplot
- [x] 4+ visualizations (histogram, boxplot, bar chart, heatmap)
- [x] Written answer on the most predictive feature
- [x] Updated notebook committed to this repo with a clear commit message
- [x] Short video (2–3 min) on one surprising visualization, posted to LinkedIn tagging Neurofive Solutions

---

**Track:** Neurofive Solutions ML Track
**Maintainer:** Huzaifa
