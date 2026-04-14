# %% [markdown]
# # Flight Delay Project Pipeline (V4) - 50k rows
#
# This notebook-style script builds and evaluates three machine learning models
# for flight delay prediction:
# - Random Forest
# - Logistic Regression
# - Gradient Boosting
#
# The target is `ARR_DEL15`, which indicates whether a flight arrived at least
# 15 minutes late.
#
# In this workflow, I:
# 1. Load and combine yearly merged CSV files.
# 2. Inspect the dataset and missing values.
# 3. Engineer time-based features.
# 4. Remove leakage variables and prepare modeling features.
# 5. Build preprocessing pipelines for numeric and categorical features.
# 6. Train and evaluate Random Forest, Logistic Regression, and Gradient Boosting.
# 7. Compare threshold behavior and save results to CSV.

# %%
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# %% [markdown]
# ## 1. Import libraries
#
# In this section, I import the libraries used throughout the project.
#
# - `pandas` and `numpy` are used for data handling and numerical work.
# - `Path` helps manage file paths cleanly.
# - `scikit-learn` provides preprocessing, modeling, and evaluation tools.

# %% [markdown]
# ## 2. Configure file paths and constants
#
# Here I define:
# - the folder containing the merged CSV files
# - the target variable (`ARR_DEL15`)
# - the random seed for reproducibility
#
# Keeping these values together makes it easier to rerun the workflow on either
# the small sample folder or the full dataset later.
# %%
DATA_DIR = Path("D:/School/GATech/CSE6242/Project/Sample_50k rows")
TARGET = "ARR_DEL15"
RANDOM_STATE = 42

# %% [markdown]
# ## 3. Load and combine the yearly merged files
#
# This function searches the data directory for all files matching
# `MERGED_*.csv`, reads them into pandas DataFrames, adds a `SOURCE_FILE`
# column for traceability, and concatenates them into one combined dataset.
#
# This lets me treat multiple yearly files as one modeling table.
# %%
def load_merged_files(data_dir: Path) -> pd.DataFrame:
    # Find all yearly merged CSV files in the target folder.
    csv_files = sorted(data_dir.glob("MERGED_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No files matching MERGED_*.csv found in: {data_dir}")

    print("Found files:")
    for f in csv_files:
        print(f" - {f.name}")

    # Store each yearly DataFrame here before concatenation.
    dataframes = []
    for f in csv_files:
        print(f"Reading {f.name} ...")
        # Skip malformed CSV rows so a single bad line does not crash the full pipeline.
        try:
            df = pd.read_csv(f, low_memory=False, on_bad_lines="skip")
            df["SOURCE_FILE"] = f.name
            dataframes.append(df)
        except Exception as e:
            print(f"Failed reading {f.name}: {e}")
            continue

    # Concatenate all successfully read files into one modeling dataset.
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nCombined shape: {combined_df.shape}")
    return combined_df

# %% [markdown]
# ## 4. Inspect the raw combined dataset
#
# This inspection step helps me verify:
# - dataset shape
# - column types
# - missing values
# - class balance for the target variable
#
# This is important before doing any feature engineering or modeling.
# %%
def inspect_dataframe(df: pd.DataFrame) -> None:
    # Show schema, row counts, and data types.
    print("\n--- BASIC INFO ---")
    print(df.info())

    print("\n--- HEAD ---")
    print(df.head())

    # Highlight the columns with the most missing data.
    print("\n--- MISSING VALUES (top 25) ---")
    print(df.isna().sum().sort_values(ascending=False).head(25))

    if TARGET in df.columns:
        # Check whether the delay target is imbalanced.
        print(f"\n--- TARGET DISTRIBUTION: {TARGET} ---")
        print(df[TARGET].value_counts(dropna=False, normalize=False))
        print("\nProportions:")
        print(df[TARGET].value_counts(dropna=False, normalize=True))

# %% [markdown]
# ## 5. Engineer features from time and date fields
#
# In this section, I create reusable helper functions to transform raw fields
# into modeling features.
#
# Examples:
# - convert HHMM scheduled times into hour-of-day values
# - parse the flight date column into a proper datetime
# - extract day-of-month (`FL_DAY`)
#
# These engineered variables help the models learn timing-related patterns.
# %%
def convert_hhmm_to_hour(value):
    if pd.isna(value):
        return np.nan

    try:
        value = int(value)
    except (ValueError, TypeError):
        return np.nan

    # Split a HHMM-style integer into hour and minute components.
    hour = value // 100
    minute = value % 100

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour
    return np.nan


def parse_flight_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # This handles values like "4/21/2015 12:00:00 AM".
    parsed = pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

    # Fallback parser for any unexpected date strings.
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(s[mask], errors="coerce")

    return parsed


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "FL_DATE" in df.columns:
        # Parse the raw flight date text into a pandas datetime.
        df["FL_DATE"] = parse_flight_date(df["FL_DATE"])
        # Extract day-of-month as a simple calendar feature.
        df["FL_DAY"] = df["FL_DATE"].dt.day

    if "CRS_DEP_TIME" in df.columns:
        # Convert scheduled departure time into hour-of-day.
        df["CRS_DEP_HOUR"] = df["CRS_DEP_TIME"].apply(convert_hhmm_to_hour)

    if "CRS_ARR_TIME" in df.columns:
        # Convert scheduled arrival time into hour-of-day.
        df["CRS_ARR_HOUR"] = df["CRS_ARR_TIME"].apply(convert_hhmm_to_hour)

    print("\n--- DATE CHECK ---")
    print(df[["FL_DATE", "FL_DAY"]].head())
    print("FL_DATE missing:", df["FL_DATE"].isna().sum())
    print("FL_DAY missing :", df["FL_DAY"].isna().sum())

    return df

# %% [markdown]
# ## 6. Filter records and select modeling features
#
# Here I prepare the supervised learning dataset.
#
# Key steps:
# - remove cancelled flights
# - keep only rows with a valid target value
# - cast the target to integer
# - remove leakage variables such as post-outcome delay breakdown fields
# - keep only the features intended for modeling
#
# This is where I make the dataset valid for prediction rather than hindsight.
# %%
def prepare_model_data(df: pd.DataFrame):
    df = df.copy()

    # Remove cancelled flights so the target focuses on completed arrivals.
    if "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0].copy()

    # Keep only rows where the prediction target is available.
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)

    # These columns would leak outcome information into the model and must be excluded.
    leakage_cols = [
        "ARR_DELAY",
        "DEP_DELAY",
        "DEP_DEL15",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "CANCELLATION_CODE",
        "TAXI_OUT",
        "TAXI_IN",
    ]

    # These columns are identifiers or raw fields that are not needed directly.
    id_like_cols = [
        "OP_CARRIER_FL_NUM",
        "ORIGIN_AIRPORT_ID",
        "DEST_AIRPORT_ID",
        "SOURCE_FILE",
        "FL_DATE",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
    ]

    exclude_cols = set(leakage_cols + id_like_cols + [TARGET])

    # These are the planned predictor variables used by the model.
    desired_features = [
        "MONTH",
        "DAY_OF_WEEK",
        "FL_DAY",
        "CRS_DEP_HOUR",
        "CRS_ARR_HOUR",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "ORIGIN_STATE_ABR",
        "DEST_STATE_ABR",
        "ORIGIN_Air_Carrier",
        "ORIGIN_Air_Taxi",
        "ORIGIN_General_Aviation",
        "ORIGIN_Military_Itinerant",
        "ORIGIN_Total_Itinerant",
        "ORIGIN_Local_Civil",
        "ORIGIN_Local_Military",
        "ORIGIN_Local_Total",
        "ORIGIN_Total_Operations",
        "DEST_Air_Carrier",
        "DEST_Air_Taxi",
        "DEST_General_Aviation",
        "DEST_Military_Itinerant",
        "DEST_Total_Itinerant",
        "DEST_Local_Civil",
        "DEST_Local_Military",
        "DEST_Local_Total",
        "DEST_Total_Operations",
        "ORIGIN_TMAX",
        "ORIGIN_TMIN",
        "ORIGIN_TAVG",
        "ORIGIN_PRCP",
        "ORIGIN_SNWD",
        "ORIGIN_AWND",
        "ORIGIN_WSF2",
        "DEST_TMAX",
        "DEST_TMIN",
        "DEST_TAVG",
        "DEST_PRCP",
        "DEST_SNWD",
        "DEST_AWND",
        "DEST_WSF2",
    ]

    feature_cols = [c for c in desired_features if c in df.columns and c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Drop any feature that ended up entirely missing after preprocessing steps.
    all_null_cols = [col for col in X.columns if X[col].isna().all()]
    if all_null_cols:
        print("\nDropping all-null columns:")
        for col in all_null_cols:
            print(f" - {col}")
        X = X.drop(columns=all_null_cols)

    print("\nPrepared modeling dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Feature columns:")
    for col in X.columns:
        print(f" - {col}")

    return X, y

# %% [markdown]
# ## 7. Build the preprocessing pipeline
#
# The models need numerical and categorical features handled differently.
#
# In this preprocessing step:
# - numeric columns use median imputation
# - categorical columns use most-frequent imputation plus one-hot encoding
#
# This ensures the data is in a usable form for all three models.
#
# Note:
# `sparse_output=False` is used so that Gradient Boosting can consume the
# transformed feature matrix without issues.
# %%
def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Separate categorical and numeric columns so they can be transformed differently.
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    print("\nCategorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    # For numeric features, fill missing values with the median.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # For categorical features, fill missing values and one-hot encode categories.
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor

# %% [markdown]
# ## 8. Define the machine learning models
#
# I train three classification models:
#
# - **Random Forest**: captures nonlinear relationships and interactions
# - **Logistic Regression**: provides a simpler linear baseline
# - **Gradient Boosting**: provides a boosting-based nonlinear model for comparison
#
# Comparing them helps me understand how much benefit comes from more flexible
# nonlinear models relative to a linear baseline.
# %%
def build_random_forest_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = get_preprocessor(X)

    # Random Forest can model nonlinear interactions between scheduling, congestion, and weather features.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline


def build_logistic_regression_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = get_preprocessor(X)

    # Logistic Regression provides a simpler linear baseline for comparison.
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline


def build_gradient_boosting_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = get_preprocessor(X)

    # HistGradientBoosting is a boosting-based model that often performs well
    # on structured/tabular datasets.
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline

# %% [markdown]
# ## 9. Evaluate probability thresholds
#
# Classification models output probabilities, not just labels.
#
# This helper function evaluates several decision thresholds (0.50, 0.40,
# 0.35, 0.30) so I can see the tradeoff between:
# - precision
# - recall
# - F1 score
# - accuracy
#
# This is especially important because flight delays are an imbalanced class.
# %%
def evaluate_at_thresholds(
    y_test: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    thresholds=None
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.50, 0.40, 0.35, 0.30]

    rows = []
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        rows.append({
            "model": model_name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, y_pred_thresh),
            "precision": precision_score(y_test, y_pred_thresh, zero_division=0),
            "recall": recall_score(y_test, y_pred_thresh, zero_division=0),
            "f1": f1_score(y_test, y_pred_thresh, zero_division=0),
        })

    results_df = pd.DataFrame(rows)
    print(f"\n--- THRESHOLD COMPARISON: {model_name} ---")
    print(results_df.to_string(index=False))
    return results_df

# %% [markdown]
# ## 10. Train and evaluate each model
#
# This function fits a model, generates predicted probabilities, applies the
# default 0.50 threshold, prints core evaluation metrics, and then compares
# performance across alternative thresholds.
#
# The evaluation includes:
# - accuracy
# - precision
# - recall
# - F1 score
# - ROC-AUC
# - confusion matrix
# - classification report
# %%
def train_and_evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str
):
    print(f"\n==============================")
    print(f"Training {model_name} ...")
    print(f"==============================")

    # Fit the selected model on the training data.
    pipeline.fit(X_train, y_train)

    # Predict class probabilities for the positive class (delay >= 15 minutes).
    print("Generating predictions ...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Convert probabilities to class labels using the default threshold of 0.50.
    y_pred = (y_proba >= 0.50).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n--- METRICS @ THRESHOLD 0.50: {model_name} ---")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)
    print("ROC-AUC  :", roc_auc)

    print(f"\n--- CONFUSION MATRIX @ THRESHOLD 0.50: {model_name} ---")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n--- CLASSIFICATION REPORT @ THRESHOLD 0.50: {model_name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    threshold_results = evaluate_at_thresholds(y_test, y_proba, model_name=model_name)

    # Store a compact summary row for side-by-side model comparison.
    summary = pd.DataFrame([{
        "model": model_name,
        "accuracy_at_0_50": acc,
        "precision_at_0_50": prec,
        "recall_at_0_50": rec,
        "f1_at_0_50": f1,
        "roc_auc": roc_auc
    }])

    return pipeline, y_proba, threshold_results, summary

# %% [markdown]
# ## 11. Interpret model behavior
#
# After training, I inspect what each model learned.
#
# - For Random Forest, I extract feature importances.
# - For Logistic Regression, I extract coefficients.
# - For Gradient Boosting, I focus on predictive performance and threshold behavior.
#
# This helps connect model performance back to the underlying drivers of delay,
# such as schedule timing, congestion, and weather.
# %%
def get_random_forest_feature_importance(pipeline: Pipeline, top_n: int = 25) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(f"\n--- TOP {top_n} RANDOM FOREST FEATURE IMPORTANCES ---")
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df


def get_logistic_regression_coefficients(pipeline: Pipeline, top_n: int = 25) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients)
    }).sort_values("abs_coefficient", ascending=False)

    print(f"\n--- TOP {top_n} LOGISTIC REGRESSION COEFFICIENTS ---")
    print(coef_df.head(top_n).to_string(index=False))

    return coef_df

# %% [markdown]
# ## 12. Run the full workflow
#
# This final section executes the pipeline end-to-end:
# 1. load data
# 2. inspect it
# 3. engineer features
# 4. prepare the modeling matrix
# 5. split into train and test sets
# 6. train all three models
# 7. save outputs to CSV
# 8. print the final side-by-side comparison
# %%
def main():
    # Load and inspect the merged airline dataset.
    df = load_merged_files(DATA_DIR)
    inspect_dataframe(df)

    # Create engineered date/time features.
    df = engineer_features(df)
    X, y = prepare_model_data(df)

    # Split into train and test sets while preserving the class ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain/Test split:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # Train and interpret the Random Forest model.
    rf_pipeline = build_random_forest_pipeline(X_train)
    rf_pipeline, rf_y_proba, rf_threshold_results, rf_summary = train_and_evaluate_model(
        pipeline=rf_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="Random Forest"
    )
    rf_importance_df = get_random_forest_feature_importance(rf_pipeline, top_n=30)

    # Train and interpret the Logistic Regression baseline.
    lr_pipeline = build_logistic_regression_pipeline(X_train)
    lr_pipeline, lr_y_proba, lr_threshold_results, lr_summary = train_and_evaluate_model(
        pipeline=lr_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="Logistic Regression"
    )
    lr_coef_df = get_logistic_regression_coefficients(lr_pipeline, top_n=30)

    # Train and evaluate the Gradient Boosting model.
    gb_pipeline = build_gradient_boosting_pipeline(X_train)
    gb_pipeline, gb_y_proba, gb_threshold_results, gb_summary = train_and_evaluate_model(
        pipeline=gb_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="Gradient Boosting"
    )

    import joblib

    # Save the full trained pipelines (preprocessor + model together)
    OUTPUT_DIR = Path("D:/School/GATech/CSE6242/Project/Sample_50k rows")
    joblib.dump(gb_pipeline, OUTPUT_DIR / "gb_pipeline_50k.pkl")
    joblib.dump(rf_pipeline, OUTPUT_DIR / "rf_pipeline_50k.pkl")
    print("Saved gb_pipeline_50k.pkl")
    print("Saved rf_pipeline_50k.pkl")

    # Save all output tables so they can be used in the final report or Tableau/D3 workflow.
    rf_importance_df.to_csv("rf_feature_importances_v4.csv", index=False)
    lr_coef_df.to_csv("lr_coefficients_v4.csv", index=False)

    all_threshold_results = pd.concat(
        [rf_threshold_results, lr_threshold_results, gb_threshold_results],
        ignore_index=True
    )
    all_threshold_results.to_csv("model_threshold_results_v4.csv", index=False)

    model_comparison = pd.concat(
        [rf_summary, lr_summary, gb_summary],
        ignore_index=True
    )
    model_comparison.to_csv("model_comparison_summary_v4.csv", index=False)

    print("\nSaved Random Forest feature importances to rf_feature_importances_v4.csv")
    print("Saved Logistic Regression coefficients to lr_coefficients_v4.csv")
    print("Saved threshold comparison to model_threshold_results_v4.csv")
    print("Saved model comparison summary to model_comparison_summary_v4.csv")

    print("\n--- FINAL MODEL COMPARISON ---")
    print(model_comparison.to_string(index=False))


if __name__ == "__main__":
    main()