"""
Adapter script to run unimol_tools using a precomputed SDF of conformers without modifying the
installed unimol_tools package.

Usage examples (zsh):

# Train using a CSV dataset and a precomputed SDF that contains conformers for the dataset
python scripts/adapter_precomputed_sdf.py --mode train --data /path/to/train.csv --sdf /path/to/conformers.sdf --save-path ./exp_run

# Predict using an existing trained model and a precomputed SDF for the test set
python scripts/adapter_precomputed_sdf.py --mode predict --data /path/to/test.csv --sdf /path/to/test_confs.sdf --load-model ./exp_run

Notes:
- This script sets the config key `precomputed_sdf` which `DataHub` in the installed unimol_tools
  package will use to populate `data['mols']` and avoid generating conformers.
- The mapping from SDF molecules to input rows is attempted by SMILES first (if present in both),
  then by index if lengths match. See unimol_tools.data.datahub.DataHub for details.

"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools # type: ignore
from rdkit import Chem
import subprocess
import tarfile
import tempfile
import shutil
try:
    import requests
except Exception:
    requests = None
try:
    import yaml
except Exception:
    yaml = None


def main():
    parser = argparse.ArgumentParser(description="Run unimol_tools end-to-end with precomputed SDF conformers: split, train, test")
    parser.add_argument('--data', required=True, help='Path to CSV dataset or dataset identifier accepted by unimol_tools')
    parser.add_argument('--sdf', required=True, help='Path to precomputed SDF containing conformers')
    parser.add_argument('--save-path', default='./exp', help='Directory to save outputs (training) or predict outputs')
    parser.add_argument('--load-model', default=None, help='Directory of trained model (required for predict)')
    # optional passthrough args for MolTrain constructor
    parser.add_argument('--task', default='classification', help='Task type: classification/regression/...')
    parser.add_argument('--model-name', default='unimolv2', help='Model name, e.g. unimolv1 or unimolv2')
    parser.add_argument('--conf-cache-level', type=int, default=1, help='conformer cache level (0/1/2)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds for cross validation (kfold)')
    parser.add_argument('--split', default='random', help='Split method: random, scaffold, group, stratified, select')
    parser.add_argument('--metrics', default='none', help="Metrics to evaluate during training/predict (use 'none' to skip)")
    parser.add_argument('--map-by', choices=['auto', 'smiles', 'index'], default='auto', help='How to map SDF entries to dataset rows: auto (try SMILES then index), smiles (require SMILES), index (require same length)')
    parser.add_argument('--smi-col', default='SMILES', help='Name of the SMILES column when mapping by SMILES')
    # splitting fractions (scaffold split will be used when --split scaffold)
    parser.add_argument('--train-frac', type=float, default=0.8, help='Fraction for training set')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Fraction for validation set (created but ignored)')
    parser.add_argument('--test-frac', type=float, default=0.1, help='Fraction for test set')

    parser.add_argument('--config', default=None, help='Path to a YAML experiment config that lists structures and urls')
    args, unknown = parser.parse_known_args()

    # If a YAML config is provided, orchestrate per-structure runs by downloading/extracting
    # and re-invoking this script for each structure. We do this via subprocess to avoid
    # refactoring the core logic.
    if args.config:
        if yaml is None:
            print('PyYAML is required to use --config; please install pyyaml')
            sys.exit(1)
        if requests is None:
            print('requests is required to use --config; please install requests')
            sys.exit(1)

        with open(args.config, 'r') as fh:
            cfg = yaml.safe_load(fh)

        file_name = None
        if isinstance(cfg, dict) and 'config' in cfg and isinstance(cfg['config'], dict):
            file_name = cfg['config'].get('file_name', None)

        structures = cfg.get('structures', {}) if isinstance(cfg, dict) else {}
        base_tmp = os.path.join(tempfile.gettempdir(), 'adapter_download')
        os.makedirs(base_tmp, exist_ok=True)

        def download_and_extract(url, out_dir):
            os.makedirs(out_dir, exist_ok=True)
            local_tar = os.path.join(out_dir, os.path.basename(url))
            if not os.path.exists(local_tar):
                print(f'Downloading {url} -> {local_tar}')
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

        def find_file(root, name):
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if f == name:
                        return os.path.join(dirpath, f)
            return None

        # iterate structures and call this script for each
        for struct_name, struct_val in structures.items():
            url = None
            if isinstance(struct_val, dict):
                url = struct_val.get('sdf_url')
            elif isinstance(struct_val, str):
                url = struct_val
            if not url:
                print(f'No URL for structure {struct_name}, skipping')
                continue
            out_dir = os.path.join(base_tmp, struct_name)
            extract_root = download_and_extract(url, out_dir)

            # expected filenames
            if file_name:
                expected_sdf = f'{file_name}.sdf'
                expected_csv = f'{file_name}.sdf.csv'
            else:
                # fallback: try to find any .sdf and .csv
                expected_sdf = None
                expected_csv = None

            sdf_path = find_file(extract_root, expected_sdf) if expected_sdf else None
            csv_path = find_file(extract_root, expected_csv) if expected_csv else None
            if sdf_path is None or csv_path is None:
                # try generic search
                for dirpath, _, files in os.walk(extract_root):
                    for f in files:
                        if f.endswith('.sdf') and sdf_path is None:
                            sdf_path = os.path.join(dirpath, f)
                        if f.endswith('.csv') and csv_path is None:
                            csv_path = os.path.join(dirpath, f)
                if sdf_path is None or csv_path is None:
                    print(f'Could not find sdf/csv pair under {extract_root} for structure {struct_name}')
                    continue

            # build subprocess args: preserve relevant flags but remove --config to avoid recursion
            cmd = [sys.executable, os.path.abspath(__file__), '--data', csv_path, '--sdf', sdf_path, '--save-path', os.path.join(args.save_path, struct_name), '--map-by', 'index']
            # preserve common flags
            for name in ['--kfold', '--split', '--task', '--metrics', '--smi-col', '--train-frac', '--val-frac', '--test-frac', '--epochs', '--model-name', '--conf-cache-level']:
                val = getattr(args, name.lstrip('-').replace('-', '_'), None)
                if val is not None:
                    cmd.extend([name, str(val)])

            print('Running:', ' '.join(cmd))
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f'Run for structure {struct_name} failed with exit code {proc.returncode}')

        # finished orchestration
        sys.exit(0)

    # Defer imports until runtime so script can be used even if unimol_tools isn't importable at parse time.
    # Prepare input_data: if an SDF is provided, load it and merge ROMol into the input data structure
    input_data = args.data
    input_df = None
    target_cols_detected = None
    if args.sdf:
        # Load SDF into a DataFrame (ROMol column) but prefer the molecule _Name as the SMILES/name.
        def load_sdf_with_names(sdf_path):
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mols = []
            names = []
            for mol in suppl:
                if mol is None:
                    mols.append(None)
                    names.append('')
                    continue
                mols.append(mol)
                name = ''
                try:
                    if mol.HasProp('_Name'):
                        name = mol.GetProp('_Name')
                except Exception:
                    name = ''
                if not name:
                    try:
                        name = Chem.MolToSmiles(mol)
                    except Exception:
                        name = ''
                names.append(name)

            df = pd.DataFrame({'SMILES': names, 'ROMol': mols})
            return df

        try:
            sdf_df = load_sdf_with_names(args.sdf)
        except Exception as e:
            print(f"Failed to load SDF {args.sdf}: {e}")
            sys.exit(1)

        # If CSV path provided, load and merge
        if isinstance(args.data, str) and args.data.endswith('.csv'):
            csv_df = pd.read_csv(args.data)
            # Detect target columns: for ESOL-like files every column is a target.
            # Exclude SMILES-like columns if present.
            cols = list(csv_df.columns)
            target_cols_detected = [c for c in cols if c not in [args.smi_col, 'SMILES', 'ROMol']]
            input_df = csv_df
            # mapping strategy
            # Mapping strategies:
            # - 'smiles': merge by SMILES column in CSV and the SMILES extracted from the SDF (_Name)
            # - 'index': require equal lengths and attach ROMol and SMILES from SDF by index
            # - 'auto': try smiles then fallback to index
            merged = None
            if args.map_by == 'smiles' or (args.map_by == 'auto' and args.smi_col in csv_df.columns and 'SMILES' in sdf_df.columns):
                merged = pd.merge(csv_df, sdf_df[['SMILES', 'ROMol']], left_on=args.smi_col, right_on='SMILES', how='left')
            elif args.map_by == 'index' or (args.map_by == 'auto' and len(csv_df) == len(sdf_df)):
                # map strictly by index: lengths must match
                if len(csv_df) != len(sdf_df):
                    print('Index mapping requested but CSV and SDF lengths differ; aborting')
                    sys.exit(1)
                csv_df = csv_df.copy()
                csv_df['ROMol'] = sdf_df['ROMol'].tolist()
                # always attach a SMILES column from SDF (from _Name) so downstream code can use it
                csv_df[args.smi_col] = sdf_df['SMILES'].tolist()
                merged = csv_df
            else:
                print('Could not map SDF to CSV: mapping method failed (try --map-by index or provide matching SMILES)')
                sys.exit(1)
            input_df = merged
            input_data = merged.to_dict(orient='list')
        elif isinstance(args.data, str) and args.data.endswith('.sdf'):
            # If input is SDF path, just use the SDF contents (romol already present)
            merged_df = PandasTools.LoadSDF(args.data)
            input_df = merged_df
            # if additional SDF provided, try to align ROMol — prefer the provided SDF
            if len(merged_df) == len(sdf_df):
                merged_df['ROMol'] = sdf_df['ROMol'].tolist()
            input_data = merged_df.to_dict(orient='list')
        elif isinstance(args.data, dict):
            # convert dict to DataFrame then attach ROMol by SMILES or index
            df = pd.DataFrame(args.data)
            # Detect target columns in dict input and preserve original names
            dict_cols = list(df.columns)
            target_cols_detected = [c for c in dict_cols if c not in [args.smi_col, 'SMILES', 'ROMol']]
            input_df = df
            # handle mapping for provided dict similarly to CSV
            if args.map_by == 'smiles' or (args.map_by == 'auto' and args.smi_col in df.columns and 'SMILES' in sdf_df.columns):
                df = pd.merge(df, sdf_df[['SMILES', 'ROMol']], left_on=args.smi_col, right_on='SMILES', how='left')
            elif args.map_by == 'index' or (args.map_by == 'auto' and len(df) == len(sdf_df)):
                if len(df) != len(sdf_df):
                    print('Index mapping requested but provided dict and SDF lengths differ; aborting')
                    sys.exit(1)
                df['ROMol'] = sdf_df['ROMol'].tolist()
                df[args.smi_col] = sdf_df['SMILES'].tolist()
            else:
                print('Could not map SDF to provided dict data')
                sys.exit(1)
            input_df = df
            input_data = df.to_dict(orient='list')
        else:
            # unknown data type; we'll try to pass sdf alone as input
            input_df = sdf_df
            input_data = sdf_df.to_dict(orient='list')

    # --- scaffold split helper (adapted from your utils.py) ---
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def generate_scaffold(smiles, include_chirality=False):
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
        except Exception:
            return None

    def scaffold_split_from_smiles(smiles_list, frac_train, frac_val, frac_test):
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
        # sort scaffold sets largest first
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
    # --- end scaffold helper ---

    # Note: this adapter now assumes a metric is always provided by the user.
    # If 'none' is passed, fail fast rather than attempting automatic selection.
    def _require_metric(metrics_arg):
        if metrics_arg is None or metrics_arg == '' or metrics_arg == 'none':
            raise ValueError("adapter requires a metric string via --metrics (do not pass 'none')")
        return metrics_arg

    # Run end-to-end: create splits, train on train split, test on test split
    from unimol_tools.train import MolTrain

    # Ensure a metric was provided
    args.metrics = _require_metric(args.metrics)

    # Restrict unimol_tools metrics registry at runtime to avoid metrics that require
    # multiple class labels on single-label datasets (best-effort, non-invasive)
    try:
        from unimol_tools import utils as um_utils

        allowed_keys = [k.strip() for k in str(args.metrics).split(',') if k.strip()]
        if allowed_keys:
            if hasattr(um_utils, 'metrics'):
                mod = um_utils.metrics
            else:
                import unimol_tools.utils.metrics as mod
            if args.task in mod.METRICS_REGISTER:
                orig = mod.METRICS_REGISTER[args.task]
                new = {k: orig[k] for k in allowed_keys if k in orig}
                if new:
                    mod.METRICS_REGISTER[args.task] = new
                    mod.DEFAULT_METRICS[args.task] = allowed_keys
    except Exception:
        pass

    trainer = MolTrain(
        task=args.task,
        model_name=args.model_name,
        save_path=args.save_path,
        conf_cache_level=args.conf_cache_level,
        epochs=args.epochs,
    kfold=1,  # force single-fold to use our supplied train set
        split=args.split,
        metrics=args.metrics,
    target_cols=target_cols_detected,
    )

    # Prepare splits and run train->test
    # We require the input as a DataFrame to split; convert if necessary
    if input_df is None:
        try:
            input_df = pd.DataFrame(input_data)
        except Exception:
            raise ValueError('Input data must be convertible to a DataFrame for splitting')

    # prefer scaffold split when requested
    if args.split == 'scaffold':
        train_idx, val_idx, test_idx = scaffold_split_from_smiles(input_df[args.smi_col].astype(str).tolist(), args.train_frac, args.val_frac, args.test_frac)
        # fallback: if scaffold split failed to produce a train set, use random split
        if len(train_idx) == 0:
            print('Scaffold split failed to produce a training set; falling back to random split')
            args.split = 'random'

    if args.split != 'scaffold':
        # simple random split
        N = len(input_df)
        rng = np.random.default_rng(42)
        idxs = rng.permutation(N)
        n_train = int(round(args.train_frac * N))
        n_val = int(round(args.val_frac * N))
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train + n_val]
        test_idx = idxs[n_train + n_val: n_train + n_val + int(round(args.test_frac * N))]

    # Build per-split dicts
    train_df = input_df.iloc[train_idx].reset_index(drop=True)
    val_df = input_df.iloc[val_idx].reset_index(drop=True) if len(val_idx) > 0 else pd.DataFrame()
    test_df = input_df.iloc[test_idx].reset_index(drop=True)

    # convert to dicts for unimol_tools (lists per column)
    train_data = train_df.to_dict(orient='list')
    test_data = test_df.to_dict(orient='list')

    print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    trainer.fit(train_data)
    print(f"Training finished. Outputs written to: {args.save_path}")

    # After training, run prediction on the test set
    from unimol_tools.predict import MolPredict
    predictor = MolPredict(load_model=args.save_path)
    print(f"Starting predict on test set. Test size: {len(test_df)}")
    y = predictor.predict(test_data, save_path=os.path.join(args.save_path, 'predict'), metrics=args.metrics)
    print(f"Predict finished. Results written to: {os.path.join(args.save_path, 'predict')}")


if __name__ == '__main__':
    main()
