# neurofive-ml-track

Task submissions for the **Neurofive Solutions ML Internship Track** — a series of hands-on tasks building ML fundamentals from the ground up, documented publicly on GitHub and LinkedIn.

---

## Task 1 — Environment Setup & First Exploratory Data Analysis (EDA)

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

**Track:** Neurofive Solutions ML Track
**Maintainer:** Huzaifa
