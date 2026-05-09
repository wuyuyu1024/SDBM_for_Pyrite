# Interpreting Pyrite Genesis With Decision Maps

This repository contains the reproducibility workflow for the manuscript **"Interpreting Mineral Genesis Classification with Decision Maps: A Case Study Using Pyrite Trace Elements"**.

The project applies supervised decision maps to pyrite trace-element data so that mineral-genesis classification can be inspected both as a predictive task and as an interpretable geoscience workflow. It is intended as research software: the code, data-processing steps, model search, visual analysis, and result tables are kept together so that the manuscript figures and classification results can be regenerated from the notebook.

## Research Context

Pyrite trace-element chemistry is commonly used to study ore-forming processes and mineral genesis. This workflow explores how decision maps can support that analysis by combining:

- supervised dimensionality reduction with SSNP;
- classifier comparison with cross-validation;
- decision-boundary visualization in a 2D embedding;
- inverse mapping from the decision-map space back to trace-element features;
- reproducible notebook-based figures and tables for manuscript results.

The work is relevant to geoscience-facing machine learning, interpretable classification, visual analytics, and environmental or Earth-science research software.

## Repository Contents

- `workflow.ipynb`: end-to-end notebook for data loading, preprocessing, model training, decision-map generation, evaluation, and manuscript tables.
- `dm_utils.py`: decision-map helper classes for classifier evaluation, probability-map rendering, and inverse-feature visualization.
- `ssnp2.py`: SSNP implementation adapted from the upstream SSNP research code.
- `requirements.txt`: tested Python package versions.
- `data/PyTE.csv`: pyrite trace-element dataset used by the workflow.
- `data/zzg2.xlsx`: additional data used by the notebook's custom-data plotting section.

## Environment

The workflow was tested with Python 3.10.8.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The pinned dependencies include TensorFlow 2.8, NumPy, pandas, matplotlib, seaborn, and imbalanced-learn.

## Reproducing The Results

Run the notebook from the repository root:

```bash
jupyter notebook workflow.ipynb
```

The notebook performs the full workflow:

1. Load and preprocess the pyrite trace-element data.
2. Train the SSNP projection and inverse projection.
3. Compare classifiers using stratified cross-validation.
4. Select the best model for the decision-map workflow.
5. Generate decision maps, confusion matrices, feature maps, and summary tables.

The current notebook output selects an `SVC(probability=True)` model in the classifier search and reports approximately 0.91 held-out classification accuracy in the displayed classification report.

## Notes For Reuse

- Run the notebook from the repository root so relative paths to `data/` resolve correctly.
- The notebook fixes random seeds in the TensorFlow/SSNP workflow where applicable, but exact results may still vary across TensorFlow, CUDA, and platform versions.
- The code is currently organized around reproducing the manuscript workflow rather than as a packaged Python library.
- If you adapt the workflow for another geochemical dataset, update the preprocessing section and class labels in `workflow.ipynb` before regenerating the decision maps.

## License

This repository is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.
