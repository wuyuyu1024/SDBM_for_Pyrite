# SDBM for Pyrite

Research software for reproducing the decision-map workflow from the following geoscience machine-learning paper:

| Field | Details |
| --- | --- |
| Paper | [Interpreting mineral deposit genesis classification with decision maps: A case study using pyrite trace elements](https://pubs.geoscienceworld.org/msa/ammin/article-abstract/109/12/2116/637125/Interpreting-mineral-deposit-genesis?redirectedFrom=fulltext) |
| Journal | *American Mineralogist*, 109(12), 2116-2129, 2024 |
| DOI | [10.2138/am-2023-9254](https://doi.org/10.2138/am-2023-9254) |

This repository implements a reproducible workflow that applies supervised decision maps to pyrite trace-element data. The goal is to inspect mineral-deposit genesis classification both as a predictive modeling task and as an interpretable geoscience workflow. The code, data-processing steps, model search, visual analysis, and result tables are kept together so that the paper's figures and classification results can be regenerated from the notebook.

## What the Workflow Does

- Loads and preprocesses pyrite trace-element data.
- Trains an SSNP-based supervised projection and inverse projection.
- Compares classifiers with stratified cross-validation.
- Builds decision maps for mineral-genesis classification.
- Maps locations in the 2D decision space back to trace-element feature estimates.
- Generates confusion matrices, feature maps, and paper-style summary tables.

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

## Citation

If you use this workflow, please cite:

> Wang, Y., Qiu, K., Telea, A., Hou, Z., Zhou, T., Cai, Y., Ding, Z., Yu, H., & Deng, J. (2024). Interpreting mineral deposit genesis classification with decision maps: A case study using pyrite trace elements. *American Mineralogist*, 109(12), 2116-2129. https://doi.org/10.2138/am-2023-9254

```bibtex
@article{wang2024pyriteDecisionMaps,
  title = {Interpreting mineral deposit genesis classification with decision maps: A case study using pyrite trace elements},
  author = {Wang, Yu and Qiu, Kunfeng and Telea, Alexandru and Hou, Zengqian and Zhou, Tao and Cai, Yujun and Ding, Zhenju and Yu, Huayong and Deng, Jun},
  journal = {American Mineralogist},
  volume = {109},
  number = {12},
  pages = {2116--2129},
  year = {2024},
  doi = {10.2138/am-2023-9254}
}
```

## Notes For Reuse

- Run the notebook from the repository root so relative paths to `data/` resolve correctly.
- The notebook fixes random seeds in the TensorFlow/SSNP workflow where applicable, but exact results may still vary across TensorFlow, CUDA, and platform versions.
- The code is currently organized around reproducing the paper workflow rather than as a packaged Python library.
- If you adapt the workflow for another geochemical dataset, update the preprocessing section and class labels in `workflow.ipynb` before regenerating the decision maps.

## License

This repository is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.
