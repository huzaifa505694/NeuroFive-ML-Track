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

## Week 1 · Task 3 — First Classification Model (Logistic Regression)

### Objective
This is where the EDA and cleaning work pays off — building the first machine learning model. Classification is one of the two core problem types in ML (the other being regression), and here the goal is to predict a category: whether a passenger **survived** or **did not survive**.

### Environment
| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Google Colab | Notebook environment (no local install needed) |
| pandas | Data loading and inspection |
| NumPy | Numerical operations |
| scikit-learn | Train/test split, encoding, model training, evaluation |

### Steps Performed
1. Started from the cleaned dataset produced in Task 2 (`Age` median-filled, `Embarked` mode-filled, `Cabin` converted to binary `Has_Cabin` and dropped)
2. Encoded categorical columns (`Sex`, `Embarked`) using `pd.get_dummies()` / `OneHotEncoder`
3. Split the dataset into training and test sets with `train_test_split` from scikit-learn
4. Trained a **Logistic Regression** model to predict `Survived`
5. Evaluated the model using `accuracy_score` from `sklearn.metrics`
6. Printed a confusion matrix and interpreted what it shows

### Model & Results
- **Model:** Logistic Regression (scikit-learn)
- **Features:** `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Has_Cabin` (categoricals one-hot encoded)
- **Final accuracy:** 0.8045 (~80%)
- **Confusion matrix:**
  |  | Predicted: Did Not Survive | Predicted: Survived |
  |---|---|---|
  | **Actual: Did Not Survive** | 96 (TN) | 14 (FP) |
  | **Actual: Survived** | 21 (FN) | 48 (TP) |

  The model correctly classifies most non-survivors and a solid majority of survivors. The 21 false negatives (actual survivors predicted as non-survivors) are the largest error group — a miss that accuracy alone doesn't surface, since it treats every correct/incorrect prediction the same regardless of class (recall Task 1: ~62% did not survive vs. ~38% survived).

### Gotcha: Colab Runtime Restarts Reset `df`
While working through this task, a `NameError`/missing-column issue came up on a feature (`Has_Cabin`) that had clearly been created earlier in the notebook. This wasn't a code bug — it was an **execution-order issue**:

- The low cell-execution counter (e.g. `[4]`) was the giveaway — it meant only 4 cells had run in that runtime session.
- `Has_Cabin` is created in the Task 2 cleaning cell (`df["Has_Cabin"] = df["Cabin"].notnull().astype(int)`), but that cell hadn't run in the current session — likely because Colab's runtime had restarted (or reconnected), and the Task 3 cell was run directly without re-running everything from the top.
- Pandas/Colab doesn't persist `df` across a runtime restart; only cells actually executed in the current session exist in memory, so any transformation from earlier tasks that wasn't re-run simply isn't there.

**Fix:**
1. In Colab: **Runtime → Run all** (or `Ctrl+F9`) so every cell executes top to bottom in order, including the Task 2 cleaning cell.
2. Or, without rerunning everything, just re-run the Task 2 cell containing:
   ```python
   df["Has_Cabin"] = df["Cabin"].notnull().astype(int)
   df = df.drop(columns=["Cabin"])
   ```
   before running the Task 3 feature/model cells.

**Habit going forward:** before recording anything for LinkedIn, do **Runtime → Restart session → Run all**, top to bottom, and confirm there are no errors. This is also the cleanest way to prove the notebook runs end-to-end for the video.

### Deliverables Checklist
- [x] Dataset split into training/test sets using `train_test_split`
- [x] Logistic Regression model trained to predict survival
- [x] Categorical columns encoded (`Sex`, `Embarked`) via `OneHotEncoder`/`pd.get_dummies()`
- [x] Model evaluated with `accuracy_score`
- [x] Confusion matrix printed and interpreted in writing
- [x] Code pushed to GitHub with README explaining approach and final accuracy
- [x] Short video walkthrough of model and results, posted to LinkedIn tagging Neurofive Solutions

### Repository Structure
```
neurofive-ml-track/
├── titanic_eda.ipynb         # EDA notebook for Task 1
├── titanic_cleaning.ipynb    # Cleaning & visualization notebook for Task 2
├── titanic_model.ipynb       # Classification model notebook for Task 3
├── titanic_tuning.ipynb      # Evaluation & hyperparameter tuning notebook for Task 4
├── titanic_pipeline.ipynb    # Pipeline & feature engineering notebook for Task 5
├── titanic_pipeline.joblib   # Saved fitted pipeline artifact from Task 5
├── README.md                 # This file
└── train.csv                 # Not committed — download from Kaggle (see above)
```

---

## Week 2 · Task 4 — Model Evaluation & Hyperparameter Tuning

### Objective
Accuracy can lie, especially with imbalanced data. This task evaluates the Task 3 model the way a real ML engineer would — with precision, recall, and F1 — and then systematically improves it with hyperparameter tuning instead of guessing at settings.

### Steps Performed
1. Revisited the Task 3 Logistic Regression model (same features: `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Has_Cabin`)
2. Calculated Precision, Recall, and F1-score with `sklearn.metrics.classification_report`
3. Tuned 2 hyperparameters (`C` and `class_weight`) with `GridSearchCV` (5-fold CV, scored on F1)
4. Compared the tuned model to the original in a before/after table

### Why Accuracy Alone Can Be Misleading
About 62% of passengers didn't survive, so a model that predicts "did not survive" for everyone would score ~61% accuracy without learning anything — and it would have 0% recall on survivors, missing every one. Accuracy weighs both classes equally, but the minority class (Survived) is the one that actually matters here. Precision, recall, and F1 expose class-level failures that a single accuracy number hides.

### Hyperparameter Tuning
- **Method:** `GridSearchCV`, tuning `C` (`[0.01, 0.1, 1, 10, 100]`) and `class_weight` (`[None, "balanced"]`), 5-fold CV, scored on F1
- **Best parameters:** `C=0.1`, `class_weight="balanced"`
- **Best CV F1-score:** 0.7419

### Before / After Comparison
| Metric | Baseline Model | Tuned Model | Change |
|---|---|---|---|
| Accuracy | 0.8045 | 0.7821 | -0.0223 |
| Precision | 0.7742 | 0.7027 | -0.0715 |
| Recall | 0.6957 | 0.7536 | +0.0580 |
| F1-Score | 0.7328 | 0.7273 | -0.0056 |

**What tuning improved (or didn't):** the tuned model trades a bit of accuracy and precision for meaningfully better **recall** on survivors — it catches more real survivors at the cost of a few more false positives. Overall F1 stayed roughly flat. This is a direct illustration of why accuracy alone would make the baseline look like the safer choice, when in fact the tuned model is the one that misses fewer actual survivors.

### Deliverables Checklist
- [x] Revisited the Task 3 classification model
- [x] Calculated Precision, Recall, F1-score with `classification_report`
- [x] Written explanation of why accuracy alone is misleading for imbalanced data
- [x] Tuned 2+ hyperparameters with `GridSearchCV`
- [x] Before/after comparison table
- [ ] Notebook pushed to GitHub
- [ ] Video walkthrough recorded and posted to LinkedIn, tagging Neurofive Solutions

---

## Week 3 · Task 5 — Building a Clean, Reusable Pipeline

### Objective
Professional ML code isn't a pile of notebook cells — it's a clean, reusable pipeline. This task replaces the manual preprocessing from Tasks 3–4 with a proper scikit-learn `Pipeline`, so preprocessing and modeling steps can't be applied inconsistently or leak data between train/test sets.

### Steps Performed
1. Reloaded the raw Titanic dataset and engineered two new features on top of the existing `Has_Cabin` flag:
   - `FamilySize` = `SibSp + Parch + 1`
   - `IsAlone` = `1` if `FamilySize == 1`, else `0`
2. Built a reusable `build_pipeline()` function using `ColumnTransformer`:
   - **Numerical branch:** `SimpleImputer(strategy="median")` → `StandardScaler()`
   - **Categorical branch:** `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore", drop="first")`
3. Chained the `ColumnTransformer` and a `LogisticRegression` classifier into a single `Pipeline` object
4. Split the data with a stratified `train_test_split`, then fit and evaluated the pipeline in one `.fit()` / `.predict()` call
5. Compared the pipeline's accuracy against the Task 3 manual-approach accuracy to confirm the refactor didn't change model behavior
6. Rebuilt the pipeline without `FamilySize`/`IsAlone` to test whether the engineered features actually earned their place
7. Saved the final fitted pipeline with `joblib.dump()`

### Pipeline Architecture
```python
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_feats),
    ("cat", categorical_transformer, categorical_feats)
])
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])
```
- **Numerical features:** `Age, Fare, SibSp, Parch, Pclass, Has_Cabin, FamilySize, IsAlone`
- **Categorical features:** `Sex, Embarked` (one-hot encoded)

### Pipeline vs. Manual Approach
| Approach | Accuracy |
|---|---|
| Manual (Task 3) | 0.8045 |
| Pipeline | 0.8045 |

The pipeline matches the manual approach's accuracy exactly — confirming the `ColumnTransformer` + `Pipeline` refactor changed *how reliably* preprocessing is applied, not the model's actual behavior.

**Confusion matrix (pipeline):**
|  | Predicted: Did Not Survive | Predicted: Survived |
|---|---|---|
| **Actual: Did Not Survive** | 94 (TN) | 16 (FP) |
| **Actual: Survived** | 19 (FN) | 50 (TP) |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Did Not Survive | 0.83 | 0.85 | 0.84 |
| Survived | 0.76 | 0.72 | 0.74 |

### Feature Engineering: Did `FamilySize` / `IsAlone` Help?
| Feature Set | Accuracy |
|---|---|
| Without `FamilySize` / `IsAlone` | 0.8101 |
| With `FamilySize` / `IsAlone` | 0.8045 |

No — accuracy actually dropped slightly (0.8101 → 0.8045). `FamilySize` is a linear combination of `SibSp` and `Parch`, both already in the model, so it adds redundant/collinear signal rather than new information, and `IsAlone` is a coarser version of the same thing. For a regularized linear model like Logistic Regression, that redundancy doesn't help and can slightly hurt. Testing engineered features against a baseline — trivial to do once the pipeline exists — is what catches this before it ships.

### Why Pipelines Matter
- **Prevents data leakage:** imputer/scaler statistics are learned only from `X_train`, never from `X_test`, since `.fit()` runs once on training data only
- **Prevents train/test inconsistency:** the same transformations apply to both sets automatically — no risk of forgetting a `fillna()` on one side (see the Task 3 Colab-restart bug above)
- **Reusable:** `build_pipeline()` was reused twice with two different feature sets in a couple of lines
- **Deployable:** the saved `.joblib` file is the whole preprocessing + model in one artifact — loading it and calling `.predict()` on raw new data just works

### Artifact
- Final fitted pipeline saved to `titanic_pipeline.joblib` via `joblib.dump()`

### Deliverables Checklist
- [x] Picked a dataset already used (Titanic)
- [x] Built a single Pipeline using `ColumnTransformer` — `StandardScaler` on numerical columns, `OneHotEncoder` on categorical columns
- [x] Chained the preprocessing step and the model into one pipeline object
- [x] Fit and evaluated the pipeline, confirming it matches the manual approach's accuracy (0.8045)
- [x] Created 2 new engineered features (`FamilySize`, `IsAlone`) and tested whether they improve performance
- [x] Saved final pipeline using `joblib`
- [ ] Notebook pushed to GitHub
- [ ] Video walkthrough recorded and posted to LinkedIn, tagging Neurofive Solutions

---

**Track:** Neurofive Solutions ML Track
**Maintainer:** Huzaifa
