import pandas as pd
from IPython.display import display
import joblib
import os

def describe_dataset(df: pd.DataFrame, target_column_name: str = None):
    # Shape of the dataset
    print("📊 Shape of Dataset:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Datatypes of each column
    print("\n🔎 Data types of columns:")
    display(df.dtypes)
    
    # Missing values (count + percentage)
    print("\n⚠️ Missing values per column:")
    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": df.isnull().mean() * 100,
        "missing_val": df.isnull().mean().apply(lambda x: f"{x:.4f}%"),
    })
    display(missing[missing.missing_count > 0])
    
    # Duplicate rows
    print("\n📑 Duplicate rows count:")
    print(df.duplicated().sum())
    
    # Statistical summary
    print("\n📈 Dataset summary:")
    display(df.describe(include="all").transpose())
    
    # Target variable info (if provided)
    if target_column_name and target_column_name in df.columns:
        print(f"\n🎯 Target variable info: {target_column_name}")
        display(df[target_column_name].describe())
        display(df[target_column_name].value_counts(normalize=True).to_frame("proportion"))
    else:
        print("\n⚠️ Target column not found or not provided.")
    
    return None

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def remove_columns_with_missing_values(df: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    # calculate the proportion of missing values for each column
    missing_proportion = df.isnull().mean().astype(float)
    
    # keep only columns where missing proportion is less than threshold
    cols_to_drop = missing_proportion[missing_proportion >= threshold].index.tolist()
    
    print(f"Columns removed: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)

def remove_columns_with_only_one_value(df: pd.DataFrame) -> pd.DataFrame:
    # calculate the proportion of unique values for each column
    cols_to_remove = [col for col in df.columns if df[col].nunique() == 1]
    return df.drop(columns=cols_to_remove)

def remove_rows_with_missing_values(df: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    # calculate the percentage of missing values for each row
    missing_rows_proportion = df.isnull().mean(axis=1).astype(float)
    rows_to_keep = missing_rows_proportion[missing_rows_proportion <= threshold].index.to_list()
    return df.loc[rows_to_keep]

# def categorize_encoding_strategy(df, cat_threshold=20):
#     strategies_with_cols = {
#         "one_hot": [],
#         "ordinal_target": []
#     }
#     for col in df.select_dtypes(include=['object', 'category']).columns:
#         n_unique = df[col].nunique()
#         if n_unique <= cat_threshold:
#             strategies_with_cols["one_hot"].append(col)
#         else:
#             strategies_with_cols["ordinal_target"].append(col)
#     return strategies_with_cols

def categorize_encoding_strategy(df, base_threshold=20, max_ohe_ratio=0.05):
    """
    Decide which categorical columns should use OneHot vs Ordinal encoding.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    base_threshold : int
        Minimum threshold for OneHot
    max_ohe_ratio : float
        Maximum ratio of n_unique / n_samples allowed for OneHot
    
    Returns
    -------
    dict
        {"one_hot": [...], "ordinal_target": [...]}
    """
    strategies_with_cols = {"one_hot": [], "ordinal_target": []}
    n_samples = df.shape[0]
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        n_unique = df[col].nunique(dropna=True)
        
        # condition: safe for OneHot if both
        if (n_unique <= base_threshold) or (n_unique / n_samples <= max_ohe_ratio):
            strategies_with_cols["one_hot"].append(col)
        else:
            strategies_with_cols["ordinal_target"].append(col)
    
    return strategies_with_cols


def find_leakage_columns(df: pd.DataFrame, target_col: str, corr_threshold: float = 0.8):
    leakage_keywords = [
        "pymnt", "rec", "recoveries", "last", "next",
        "out_prncp", "settlement", "hardship", "collection"
    ]

    # 1. Columns with suspicious names
    suspicious_cols = [col for col in df.columns if any(key in col.lower() for key in leakage_keywords)]

    # 2. Correlation check with target (numeric encoding)
    corr_suspicious = []
    if df[target_col].dtype == "object":
        target_numeric = df[target_col].astype("category").cat.codes
    else:
        target_numeric = df[target_col]

    for col in df.select_dtypes(include=['int64','float64']).columns:
        if col != target_col:
            corr = df[col].corr(target_numeric)
            if abs(corr) >= corr_threshold:
                corr_suspicious.append((col, corr))

    print("🚨 Suspicious by name:")
    print(suspicious_cols)
    print("\n🚨 Suspicious by correlation (>|{}|):".format(corr_threshold))
    for col, corr in corr_suspicious:
        print(f"{col}: {corr:.3f}")

    # Combine all
    leakage_cols = set(suspicious_cols + [c for c,_ in corr_suspicious])
    return list(leakage_cols)

# Example usage:
# leakage_cols = find_leakage_columns(df, target_col="loan_status")

# save the model
def save_model(pipelines:dict, folder='saved_models'):
    os.makedirs(folder, exist_ok=True)
    for pipelines_name, pipeline in pipelines.items():
        # create a path of svaed models
        model_path = os.path.join(folder, f"{pipelines_name.replace(' ', '_').lower()}.pkl")
        # save the models
        joblib.dump(pipeline, model_path)
    
    print("model is saved")
    
def load_selective_models(model_names: list, folder='saved_models'):
    models = {}
    for model_name in model_names:
        model_path = os.path.join(folder, f"{model_name.replace(' ', '_').lower()}.pkl")
        models[model_name] = joblib.load(model_path)
    return models
        
