from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Function to get fingerprints
def get_fingerprints(smiles, weights):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if not molecule:
            return None
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

    # Parameters need to be adjusted according to different problems and datasets

    fingerprint_funcs = {
        'maccs': lambda mol: MACCSkeys.GenMACCSKeys(mol),
        'ecfp': lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512),
        'morgan': lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512),
        'pubchem': lambda mol: Chem.RDKFingerprint(mol, fpSize=512),
        'fcfp': lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128, useFeatures=True),
        'fp2': lambda mol: AllChem.RDKFingerprint(mol, fpSize=512),
        'daylight': lambda mol: Chem.RDKFingerprint(mol, fpSize=512)
    }
    arrays = []
    for name, func in fingerprint_funcs.items():
        if weights[name] > 0.001:  # Only compute and include the fingerprint if its weight is greater than zero!!
            fp = func(molecule)
            arr = np.zeros((fp.GetNumBits(),), dtype=float)
            DataStructs.ConvertToNumpyArray(fp, arr)
            arr *= weights[name]
            arrays.append(arr)
    concatenated = np.concatenate(arrays) if arrays else np.array([])
    return concatenated


# Objective function for Bayesian optimization
def mse_model(maccs_weight, ecfp_weight, morgan_weight, pubchem_weight, fcfp_weight, fp2_weight, daylight_weight):
    weights = {
        'maccs': maccs_weight,
        'ecfp': ecfp_weight,
        'morgan': morgan_weight,
        'pubchem': pubchem_weight,
        'fcfp': fcfp_weight,
        'fp2': fp2_weight,
        'daylight': daylight_weight
    }
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    combined_fingerprints = []
    valid_pce = []
    for smiles, pce in zip(data['SMILES Code'], data['PCE']):
        fp = get_fingerprints(smiles, weights)
        if fp is not None:
            combined_fingerprints.append(fp)
            valid_pce.append(pce)

    X = np.array(combined_fingerprints)
    y = np.array(valid_pce)

    pca = PCA(n_components=0.8)
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5,
                              alpha=10, n_estimators=100)
    xg_reg.fit(X_train, y_train)
    y_pred_xgb = xg_reg.predict(X_test)
    return -mean_squared_error(y_test, y_pred_xgb)


# Define parameter bounds for Bayesian optimization
# Parameters need to be adjusted according to different problems and datasets
pbounds = {
    'maccs_weight': (0, 1),
    'ecfp_weight': (0, 1),
    'morgan_weight': (0, 1),
    'pubchem_weight': (0, 1),
    'fcfp_weight': (0, 1),
    'fp2_weight': (0, 1),
    'daylight_weight': (0, 1)
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=mse_model, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=10, n_iter=30)

# Print the best result
print("Best result:", optimizer.max)
