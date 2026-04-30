# CSCIS412

This repository contains a prompt-injection detection project that trains a classifier against a local prompt-injection dataset.
The default configuration now trains on the full deduplicated dataset for better coverage, while keeping rebalancing available as an option.

## Files

- `implementation.py` - Main training script. Builds the dataset, rebalances classes, trains the model, and prints evaluation results.
- `prompt_injection_data.py` - Dataset loading, normalization, balancing, and model-training helpers.
- `implementation_dynamic.py` - Standalone rule-based filter demo.
- `implementation2.0.ipynb` - Notebook version of the project.

## How to use

1. Open the project in VS Code.
2. Make sure the Kaggle CSV files exist under `data/kaggle_prompt_injection/`.
3. Run:

```powershell
& "C:\Users\Juanc\AppData\Local\Programs\Python\Python313\python.exe" implementation.py
```

4. Optional flags:

```powershell
& "C:\Users\Juanc\AppData\Local\Programs\Python\Python313\python.exe" implementation.py --interactive
& "C:\Users\Juanc\AppData\Local\Programs\Python\Python313\python.exe" implementation.py --model-out artifacts/prompt_injection_model.joblib
& "C:\Users\Juanc\AppData\Local\Programs\Python\Python313\python.exe" implementation.py --use-balanced-dataset
```
