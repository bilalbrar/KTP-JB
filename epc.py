# -*- coding: utf-8 -*-


# Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc

# File paths
certs_path = r"C:\Users\bilal\BCU\Birmingham Energy Performance\certificates.csv"
recs_path  = r"C:\Users\bilal\BCU\Birmingham Energy Performance\recommendations.csv"

# Load data
certs = pd.read_csv(certs_path,low_memory=False)
recs  = pd.read_csv(recs_path)

# EDA
certs.head()

certs.describe()

recs.head()

recs.describe()

# Basic inspection
print("Certificates:", certs.shape)
print("\nRecommendations:", recs.shape)

#— unique key counts
n_cert_keys = certs["LMK_KEY"].nunique()
n_rec_keys  = recs["LMK_KEY"].nunique()

#— how many keys are in both?
common_keys = set(certs["LMK_KEY"]).intersection(recs["LMK_KEY"])
n_common    = len(common_keys)

#— percentages
pct_certs_with_rec = n_common / n_cert_keys * 100
pct_recs_with_cert = n_common / n_rec_keys  * 100

#— recommendation‐per‐property stats
rec_counts = recs.groupby("LMK_KEY").size().loc[list(common_keys)]
print("Per-property rec counts summary:")
print(rec_counts.describe())

#— print
print(f"Unique LMK_KEY in certificates:           {n_cert_keys}")
print(f"Unique LMK_KEY in recommendations:        {n_rec_keys}")
print(f"Keys present in both (common):            {n_common}")
print(f"% of certificates having ≥1 recommendation: {pct_certs_with_rec:.1f}%")
print(f"% of recommendation-keys matching a cert:   {pct_recs_with_cert:.1f}%")

# show missing % for all certs columns in descending order
import pandas as pd

pd.set_option('display.max_rows', None)

null_counts = certs.isnull().sum()
null_pct    = (null_counts / len(certs) * 100).round(2)

missing_df = (
    pd.DataFrame({
        'column':      null_counts.index,
        'missing_pct': null_pct.values
    })
    .sort_values('missing_pct', ascending=False)
)

print(missing_df.to_string(index=False))

# Inspect recommendations
print("=== Recommendations ===")
print(f"Shape: {recs.shape}")
print("\nColumns:")
for col in recs.columns:
    print(" ", col)

nulls_r = recs.isnull().sum()
pct_r   = recs.isnull().mean() * 100
print("\nMissing values per column:")
print(pd.DataFrame({
    "null_count": nulls_r,
    "null_pct": pct_r.round(2)
}).sort_values("null_count", ascending=False))

# show count of each dtype
print(certs.dtypes.value_counts(), "\n")

# list only the object-type columns
obj_cols = certs.dtypes[certs.dtypes == 'object'].index.tolist()
print(f"{len(obj_cols)} object-type columns:\n", obj_cols)

for col in obj_cols:
    top = certs[col].value_counts(dropna=False).head(5)
    print(f"\n--- {col} (dtype={certs[col].dtype}) ---")
    print(top)

import numpy as np

# 1a) Parse dates
for dt in ["INSPECTION_DATE", "LODGEMENT_DATE", "LODGEMENT_DATETIME"]:
    certs[dt] = pd.to_datetime(certs[dt], errors="coerce")

# 1b) Unify “NO DATA!” and empty strings into NaN
placeholder_cols = certs.select_dtypes(include="object").columns
for col in placeholder_cols:
    certs[col] = certs[col].replace({"NO DATA!": np.nan, "": np.nan})

# Cell: 2) Cast categories
cat_cols = [
    "CURRENT_ENERGY_RATING","POTENTIAL_ENERGY_RATING",
    "PROPERTY_TYPE","BUILT_FORM","TRANSACTION_TYPE","ENERGY_TARIFF",
    "MAIN_FUEL","CONSTRUCTION_AGE_BAND","TENURE",
    "LOCAL_AUTHORITY","CONSTITUENCY","LOCAL_AUTHORITY_LABEL","CONSTITUENCY_LABEL",
    # flag-style cats
    "MAINS_GAS_FLAG","SOLAR_WATER_HEATING_FLAG","PHOTO_SUPPLY","LOW_ENERGY_LIGHTING",
    "FLAT_TOP_STOREY"
]
for col in cat_cols:
    if col in certs:
        certs[col] = certs[col].astype("category")

# Cell: 3) Coerce truly-numeric strings into numbers
# (Add any you spot in your audit; these are examples)
to_num = ["MULTI_GLAZE_PROPORTION","FLOOR_HEIGHT"]
for col in to_num:
    certs[col] = (
        certs[col]
        .astype(str)
        .str.replace("[^0-9.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

# Re-check dtypes
print(certs.dtypes.value_counts())

# Cell: list remaining object columns
obj_cols = certs.select_dtypes(include='object').columns.tolist()
print(f"{len(obj_cols)} object columns:\n", obj_cols)


# 1) Drop free-text, geodata, very sparse, and low-value columns
drop_cols = [
    'BUILDING_REFERENCE_NUMBER',
    # geodata labels
    'LOCAL_AUTHORITY_LABEL','CONSTITUENCY_LABEL','COUNTY',
    # long descriptions
    'HOTWATER_DESCRIPTION','FLOOR_DESCRIPTION','WINDOWS_DESCRIPTION',
    'WALLS_DESCRIPTION','ROOF_DESCRIPTION','MAINHEAT_DESCRIPTION',
    'MAINHEATCONT_DESCRIPTION','LIGHTING_DESCRIPTION','SECONDHEAT_DESCRIPTION',
    # sparse features
    'FLAT_STOREY_COUNT','UNHEATED_CORRIDOR_LENGTH','LOW_ENERGY_FIXED_LIGHT_COUNT',
    # all-NaN or low-value features
    'SHEATING_ENV_EFF','SHEATING_ENERGY_EFF',
    'FLOOR_ENV_EFF','FLOOR_ENERGY_EFF',
    'FLAT_TOP_STOREY','ADDRESS2','ADDRESS3'
]
certs.drop(columns=drop_cols, inplace=True, errors='ignore')

# 2) Parse date columns to datetime (if needed elsewhere)
date_cols = ['INSPECTION_DATE','LODGEMENT_DATE','LODGEMENT_DATETIME']
for dt in date_cols:
    if dt in certs.columns:
        certs[dt] = pd.to_datetime(certs[dt], errors='coerce')

# 3) Normalize placeholder strings to NaN
certs.replace({'NO DATA!': np.nan, '': np.nan}, inplace=True)

# 4) Map Y/N flags to binary 0/1
flag_map = {'Y': 1, 'N': 0}
for col in ['MAINS_GAS_FLAG','SOLAR_WATER_HEATING_FLAG']:
    if col in certs.columns:
        certs[col] = certs[col].map(flag_map).fillna(0).astype(int)

# 5) Coerce numeric-in-string columns into float
for col in ['MULTI_GLAZE_PROPORTION','FLOOR_HEIGHT']:
    if col in certs.columns:
        certs[col] = (
            certs[col].astype(str)
                       .str.replace(r'[^0-9.]', '', regex=True)
                       .replace('', np.nan)
                       .astype(float)
        )

# 6) Cast efficiency and rating fields to categorical
eff_cols = [
    'CURRENT_ENERGY_RATING','POTENTIAL_ENERGY_RATING',
    'HOT_WATER_ENERGY_EFF','HOT_WATER_ENV_EFF',
    'WINDOWS_ENERGY_EFF','WINDOWS_ENV_EFF',
    'WALLS_ENERGY_EFF','WALLS_ENV_EFF',
    'LIGHTING_ENERGY_EFF','LIGHTING_ENV_EFF',
    'ROOF_ENERGY_EFF','ROOF_ENV_EFF',
    'MAINHEAT_ENERGY_EFF','MAINHEAT_ENV_EFF',
    'MAINHEATC_ENERGY_EFF','MAINHEATC_ENV_EFF'
]
for col in eff_cols:
    if col in certs.columns:
        certs[col] = certs[col].astype('category')

# 7) Cast small-set code columns to category
small_cat = [
    'PROPERTY_TYPE','BUILT_FORM','TRANSACTION_TYPE','ENERGY_TARIFF',
    'MAIN_FUEL','CONSTRUCTION_AGE_BAND','TENURE',
    'LOCAL_AUTHORITY','CONSTITUENCY',
    'MECHANICAL_VENTILATION','MAIN_HEATING_CONTROLS',
    'GLAZED_TYPE','GLAZED_AREA'
]
for col in small_cat:
    if col in certs.columns:
        certs[col] = certs[col].astype('category')

# 8) Drop any remaining object-type columns except LMK_KEY, ADDRESS, ADDRESS1, POSTCODE, POSTTOWN
allowed = {'LMK_KEY','ADDRESS','ADDRESS1','POSTCODE','POSTTOWN'}
remaining_obj = [c for c in certs.select_dtypes(include='object').columns if c not in allowed]
certs.drop(columns=remaining_obj, inplace=True, errors='ignore')

# 9) Final dtype summary
print(certs.dtypes.value_counts())

# Prepare final feature lists (dropping all date columns)

# 1) Drop any datetime columns
datetime_feats = certs.select_dtypes(include=['datetime64[ns]']).columns.tolist()
certs.drop(columns=datetime_feats, inplace=True)

# 2) Assemble feature lists by dtype
numeric_feats = certs.select_dtypes(include=['float64','int64']).columns.tolist()
categorical_feats = certs.select_dtypes(include=['category']).columns.tolist()

print(f"Dropped date columns: {datetime_feats}")
print("Numeric features (count={}):".format(len(numeric_feats)), numeric_feats)
print("Categorical features (count={}):".format(len(categorical_feats)), categorical_feats)

certs.info()

# show missing % for all certs columns in descending order

pd.set_option('display.max_rows', None)

null_counts = recs.isnull().sum()
null_pct    = (null_counts / len(recs) * 100).round(2)

missing_df = (
    pd.DataFrame({
        'column':      null_counts.index,
        'missing_pct': null_pct.values
    })
    .sort_values('missing_pct', ascending=False)
)

print(missing_df.to_string(index=False))

recs.drop(columns=['IMPROVEMENT_ID_TEXT'], inplace=True, errors='ignore')

# Build final one-row-per‐LMK_KEY DataFrame with only matching keys

import pandas as pd


# 1) Identify matching LMK_KEYs
common_keys = set(certs['LMK_KEY']).intersection(recs['LMK_KEY'])
print(f"Number of matching LMK_KEYs: {len(common_keys)}")

# 2) Filter both tables to only those keys
certs_matched = certs[certs['LMK_KEY'].isin(common_keys)].copy()
recs_matched  = recs[recs['LMK_KEY'].isin(common_keys)].copy()

# 3) Aggregate recommendation features per property
recs_agg = (
    recs_matched
    .groupby('LMK_KEY')
    .agg(
        num_recommendations = ('IMPROVEMENT_ITEM', 'count'),
        improvement_ids     = ('IMPROVEMENT_ID', list),
        indicative_costs    = ('INDICATIVE_COST', list)
    )
    .reset_index()
)

# 4) Merge (inner) so only properties with both certs & recs appear
final_df = pd.merge(
    certs_matched,
    recs_agg,
    on='LMK_KEY',
    how='inner'
)

# 5) Inspect
print("Final DataFrame shape:", final_df.shape)
print("Sample columns:", final_df.columns.tolist()[:10], "…")
display(final_df.head())

# Map the target POTENTIAL_ENERGY_RATING to an ordinal numeric scale
rating_order = ['A','B','C','D','E','F','G']
rating_map = {r: i for i, r in enumerate(rating_order, start=1)}
final_df['POTENTIAL_ENERGY_RATING_NUM'] = final_df['POTENTIAL_ENERGY_RATING'].map(rating_map)

# Plot distribution of the target variable
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(
    x='POTENTIAL_ENERGY_RATING',
    data=final_df,
    order=rating_order
)
plt.title('Distribution of Potential Energy Rating')
plt.xlabel('Potential Energy Rating')
plt.ylabel('Count')
plt.show()

# Compute correlation matrix for all numeric features
numeric_cols = final_df.select_dtypes(include=['int64','float64']).columns.tolist()
# ensure our new numeric target is included
if 'POTENTIAL_ENERGY_RATING_NUM' not in numeric_cols:
    numeric_cols.append('POTENTIAL_ENERGY_RATING_NUM')

corr_matrix = final_df[numeric_cols].corr()

# Plot the correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5}
)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Check missing values in final_df

import pandas as pd

# Configure pandas to show all rows
pd.set_option('display.max_rows', None)

# Calculate missing counts and percentages
missing_counts = final_df.isnull().sum()
missing_pct = (missing_counts / len(final_df) * 100).round(2)

# Build and display a sorted DataFrame
missing_df = pd.DataFrame({
    'column': missing_counts.index,
    'missing_count': missing_counts.values,
    'missing_pct': missing_pct.values
}).sort_values('missing_pct', ascending=False)

print(missing_df.to_string(index=False))

# Simple imputation — fill missing with median or "Missing"

import pandas as pd
import numpy as np

# assume final_df is already defined

# 1) Numeric features: coerce to numeric and fill missing with median
num_cols = ['FIXED_LIGHTING_OUTLETS_COUNT', 'PHOTO_SUPPLY']
for col in num_cols:
    if col in final_df.columns:
        # ensure numeric dtype
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        med = final_df[col].median(skipna=True)
        final_df[col] = final_df[col].fillna(med)

# 2) Categorical features: add 'Missing' category and fill missing
import pandas.api.types as ptypes

cat_cols = ['MAIN_HEATING_CONTROLS', 'ROOF_ENERGY_EFF', 'ROOF_ENV_EFF']
for col in cat_cols:
    if col in final_df.columns:
        # if already categorical, add new category
        if isinstance(final_df[col].dtype, pd.CategoricalDtype):
            final_df[col] = final_df[col].cat.add_categories(['Missing'])
        # fill NaNs
        final_df[col] = final_df[col].fillna('Missing')

# 3) Drop any rows with remaining NaNs in the entire DataFrame
final_df = final_df.dropna(how='any')

# 4) Verify that no missing remain in the imputed columns
print(final_df[num_cols + cat_cols].isnull().sum())

# Model Implementation

from joblib import parallel_backend
import time  # Also adding time import for the time.time() calls
import os  # Adding missing os import

# Drop identifiers and address fields
X = final_df.drop(columns=[
    'LMK_KEY','UPRN', 'ADDRESS', 'ADDRESS1', 'POSTCODE', 'POSTTOWN',
    'POTENTIAL_ENERGY_RATING'
])
y = final_df['POTENTIAL_ENERGY_RATING']

# Encode target labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# Preprocessing pipeline
numeric_feats = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_feats = X_train.select_dtypes(include=['category']).columns.tolist()

num_pipe = Pipeline([
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_feats),
    ('cat', cat_pipe, categorical_feats)
])

# Define models - SIMPLIFIED VERSION with fewer models
models = {
    'LogisticRegression': LogisticRegression(max_iter=500, solver='saga'),  # Faster solver
    'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)  # Reduced complexity
}

# Train using SMOTE to balance minority classes
print("Training models with SMOTE to balance classes...")
results = {}

# always train on the full dataset
X_train_sample = X_train
y_train_sample = y_train

# Use parallel processing and simpler evaluation
with parallel_backend('threading', n_jobs=-1):
    for name, model in models.items():
        start_time = time.time()
        
        # pipeline with preprocessing, SMOTE oversampling, and classifier
        pipe = ImbPipeline([
            ('preproc', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', model)
        ])
        
        pipe.fit(X_train_sample, y_train_sample)
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {'pipeline': pipe, 'f1': f1}
        print(f"✓ {name} trained in {time.time() - start_time:.2f}s - F1: {f1:.4f}")

# Select best by F1
best_name = max(results, key=lambda k: results[k]['f1'])
best_pipe = results[best_name]['pipeline']
print(f"\n Best performing model: {best_name} (F1={results[best_name]['f1']:.4f})")

# Save the best model and components
print("Saving the best model and related components...")

# Use relative path for saving models
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save model and components using relative paths
import joblib
import os
joblib.dump(best_pipe, os.path.join(models_dir, "epc_best_model.pkl"))
joblib.dump(preprocessor, os.path.join(models_dir, "epc_preprocessor.pkl"))
joblib.dump(le, os.path.join(models_dir, "epc_label_encoder.pkl"))

# Also save important feature metadata for the API
feature_metadata = {
    'numeric_features': numeric_feats,
    'categorical_features': categorical_feats
}
joblib.dump(feature_metadata, os.path.join(models_dir, "epc_feature_metadata.pkl"))
print(f"Model and components saved successfully to {models_dir}")

print("Training and model saving complete!")



