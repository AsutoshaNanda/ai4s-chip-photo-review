# AI4S code directory

The primary guide is the `README.md` file in the top directory of this repository. Read that file first.

## Run the photo tool

1. Open a terminal in this `code` directory.
2. Install the declared packages.
3. Run the tool on a folder of photos.

```bash
python -m pip install -r requirements.txt
python inference.py --photos path/to/photos --out ranking.csv --sheet sheet.png
```

## Contents of this directory

| Path | Contents |
|---|---|
| `inference.py` | The entry point of the photo tool. |
| `model_x01_3x3.joblib` and `model_x01_3x3.json` | The saved scorer and its metadata. |
| `final_config.json` | The fixed choices for the final evaluations. |
| `requirements.txt` | The package versions used for the clean rebuild. |
| `src/ai4s` | Shared code. |
| `scripts` | Experiment scripts. |
| `splits` | The fixed data split files with their hashes. |
| `runs` | Saved results. |
| `kaggle` | Kaggle job files and job outputs. |
| `cloud` | Helper scripts for Modal cloud jobs. |
| `logs` | Job logs. |
| `demo` | Example outputs from one recorded run. |

The code has the Apache License 2.0. Read the `LICENSE` file in the top directory.
