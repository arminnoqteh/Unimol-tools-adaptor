"""Library utilities for adapter_precomputed_sdf CLI.

Contains functions for loading SDFs, mapping to CSVs, splitting, and running unimol_tools
training/prediction. This centralizes logic so the CLI can remain small and testable.
"""
from __future__ import annotations

import os
import tarfile
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools 


def load_sdf_with_names(sdf_path: str) -> pd.DataFrame:
    """Load an SDF and return DataFrame with columns ['SMILES','ROMol'] where SMILES
    is taken from the molecule _Name if present, otherwise from MolToSmiles.
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = []
    names = []
    for mol in suppl:
        if mol is None:
            mols.append(None)
            names.append("")
            continue
        mols.append(mol)
        name = ""
        try:
            if mol.HasProp('_Name'):
                name = mol.GetProp('_Name')
        except Exception:
            name = ""
        if not name:
            try:
                name = Chem.MolToSmiles(mol)
            except Exception:
                name = ""
        names.append(name)

    df = pd.DataFrame({'SMILES': names, 'ROMol': mols})
    return df


def map_sdf_to_df(csv_df: pd.DataFrame, sdf_df: pd.DataFrame, map_by: str = 'auto', smi_col: str = 'SMILES') -> pd.DataFrame:
    """Attach ROMol and SMILES from sdf_df to csv_df based on mapping strategy.

    map_by: 'auto'|'smiles'|'index'
    """
    merged = None
    if map_by == 'smiles' or (map_by == 'auto' and smi_col in csv_df.columns and 'SMILES' in sdf_df.columns):
        merged = pd.merge(csv_df, sdf_df[['SMILES', 'ROMol']], left_on=smi_col, right_on='SMILES', how='left')
    elif map_by == 'index' or (map_by == 'auto' and len(csv_df) == len(sdf_df)):
        if len(csv_df) != len(sdf_df):
            raise ValueError('Index mapping requested but CSV and SDF lengths differ')
        csv_df = csv_df.copy()
        csv_df['ROMol'] = sdf_df['ROMol'].tolist()
        csv_df[smi_col] = sdf_df['SMILES'].tolist()
        merged = csv_df
    else:
        raise ValueError('Could not map SDF to CSV: mapping method failed')
    return merged


def detect_target_cols(df: pd.DataFrame, smi_col: str = 'SMILES') -> List[str]:
    """Detect target columns in a DataFrame: all columns except SMILES/ROMol/RowID are treated as targets."""
    cols = [c for c in df.columns if c not in [smi_col, 'SMILES', 'ROMol', 'RowID']]
    return cols


from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str, include_chirality: bool = False) -> Optional[str]:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    except Exception:
        return None


def scaffold_split_from_smiles(smiles_list: List[str], frac_train: float, frac_val: float, frac_test: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(smiles_list)
    all_scaffolds = {}
    failed = []
    for i, smi in enumerate(smiles_list):
        scaffold = generate_scaffold(smi)
        if scaffold is None:
            failed.append(i)
            continue
        all_scaffolds.setdefault(scaffold, []).append(i)

    N = N - len(failed)
    all_scaffold_sets = [s for (_, s) in sorted(all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_val) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return np.array(train_idx, dtype=int), np.array(valid_idx, dtype=int), np.array(test_idx, dtype=int)


def download_and_extract(url: str, out_dir: str) -> str:
    """Download a tarball and extract it to out_dir/extracted, returning the extract root."""
    import requests

    os.makedirs(out_dir, exist_ok=True)
    local_tar = os.path.join(out_dir, os.path.basename(url))
    if not os.path.exists(local_tar):
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(local_tar, 'wb') as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    extract_root = os.path.join(out_dir, 'extracted')
    if not os.path.exists(extract_root):
        os.makedirs(extract_root, exist_ok=True)
        try:
            with tarfile.open(local_tar, 'r:*') as tf:
                tf.extractall(path=extract_root)
        except Exception:
            with tarfile.open(local_tar, 'r:gz') as tf:
                tf.extractall(path=extract_root)
    return extract_root


def find_file(root: str, name: Optional[str]) -> Optional[str]:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if name is not None and f == name:
                return os.path.join(dirpath, f)
            if name is None and f.endswith('.sdf'):
                return os.path.join(dirpath, f)
    return None


def process_pair(data_path: str, sdf_path: str, save_path: str, args) -> None:
    """Process a single CSV+SDF pair: map, split, train, predict.

    `args` is expected to have attributes used in the original script (map_by, smi_col, split, train_frac, val_frac, test_frac, task, model_name, conf_cache_level, epochs, metrics).
    """
    # load sdf
    sdf_df = load_sdf_with_names(sdf_path)

    # load data
    if isinstance(data_path, str) and data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif isinstance(data_path, str) and data_path.endswith('.sdf'):
        df = PandasTools.LoadSDF(data_path)
    else:
        # assume path-like string; try read_csv
        df = pd.read_csv(data_path)

    # detect targets before mapping (preserve original names)
    target_cols = detect_target_cols(df, smi_col=args.smi_col)

    # map sdf to df
    df = map_sdf_to_df(df, sdf_df, map_by=args.map_by, smi_col=args.smi_col)

    # prepare trainer kwargs
    trainer_kwargs = dict(
        task=args.task,
        model_name=args.model_name,
        save_path=save_path,
        conf_cache_level=args.conf_cache_level,
        epochs=args.epochs,
        kfold=1,
        split=args.split,
        metrics=args.metrics,
    )
    if target_cols:
        trainer_kwargs['target_cols'] = target_cols

    from unimol_tools.train import MolTrain

    trainer = MolTrain(**trainer_kwargs)

    # Splitting
    if args.split == 'scaffold':
        train_idx, val_idx, test_idx = scaffold_split_from_smiles(df[args.smi_col].astype(str).tolist(), args.train_frac, args.val_frac, args.test_frac)
        if len(train_idx) == 0:
            print('Scaffold split failed to produce a training set; falling back to random split')
            args.split = 'random'

    if args.split != 'scaffold':
        N = len(df)
        rng = np.random.default_rng(42)
        idxs = rng.permutation(N)
        n_train = int(round(args.train_frac * N))
        n_val = int(round(args.val_frac * N))
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train + n_val]
        test_idx = idxs[n_train + n_val: n_train + n_val + int(round(args.test_frac * N))]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True) if len(val_idx) > 0 else pd.DataFrame()
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_data = train_df.to_dict(orient='list')
    test_data = test_df.to_dict(orient='list')

    print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    trainer.fit(train_data)
    print(f"Training finished. Outputs written to: {save_path}")

    from unimol_tools.predict import MolPredict
    predictor = MolPredict(load_model=save_path)
    print(f"Starting predict on test set. Test size: {len(test_df)}")
    predictor.predict(test_data, save_path=os.path.join(save_path, 'predict'), metrics=args.metrics)
    print(f"Predict finished. Results written to: {os.path.join(save_path, 'predict')}")
