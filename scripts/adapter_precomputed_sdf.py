"""Small CLI wrapper for the adapter library.

This file parses CLI args and delegates work to `adapter_lib.process_pair`. When a
YAML `--config` is provided it will iterate structures, download/extract archives
and call `process_pair` for each structure.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Optional

import yaml

# When executed as a script the package-relative import `from . import adapter_lib`
# fails (no parent package). Ensure the scripts directory is on sys.path and import
# the module by filename so the CLI works both as `python scripts/adapter_precomputed_sdf.py`
# and when imported as a package.
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import adapter_lib


def main():
    parser = argparse.ArgumentParser(description='Adapter CLI')
    parser.add_argument('--data', required=False, help='CSV path or dataset identifier')
    parser.add_argument('--sdf', required=False, help='SDF path')
    parser.add_argument('--save-path', default='./exp', help='Output directory')
    parser.add_argument('--config', default=None, help='YAML config with structures and urls')
    parser.add_argument('--map-by', choices=['auto', 'smiles', 'index'], default='auto')
    parser.add_argument('--smi-col', default='SMILES')
    parser.add_argument('--split', default='random')
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--task', default='regression')
    parser.add_argument('--model-name', default='unimolv2')
    parser.add_argument('--conf-cache-level', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--metrics', default='mse')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as fh:
            cfg = yaml.safe_load(fh)
        file_name = cfg.get('config', {}).get('file_name') if isinstance(cfg, dict) else None
        structures = cfg.get('structures', {}) if isinstance(cfg, dict) else {}
        base_tmp = os.path.join(tempfile.gettempdir(), 'adapter_download')
        os.makedirs(base_tmp, exist_ok=True)

        for struct_name, struct_val in structures.items():
            url = struct_val.get('sdf_url') if isinstance(struct_val, dict) else struct_val
            if not url:
                print(f'No URL for structure {struct_name}, skipping')
                continue
            out_dir = os.path.join(base_tmp, struct_name)
            extract_root = adapter_lib.download_and_extract(url, out_dir)
            # find expected files
            sdf_path = adapter_lib.find_file(extract_root, f'{file_name}.sdf' if file_name else None)
            csv_path = adapter_lib.find_file(extract_root, f'{file_name}.sdf.csv' if file_name else None)
            if not sdf_path or not csv_path:
                print(f'Could not find sdf/csv for {struct_name} under {extract_root}, skipping')
                continue
            save_path = os.path.join(args.save_path, struct_name)
            adapter_lib.process_pair(csv_path, sdf_path, save_path, args)
    else:
        if not args.data or not args.sdf:
            print('Either --config or both --data and --sdf must be provided')
            sys.exit(1)
        os.makedirs(args.save_path, exist_ok=True)
        adapter_lib.process_pair(args.data, args.sdf, args.save_path, args)


if __name__ == '__main__':
    main()
