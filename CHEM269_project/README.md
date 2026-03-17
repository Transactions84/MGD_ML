# Predictive Modeling of MGD-Mediated ZBTB11 and IKZF1 Degradation
## Project Summary

This project implements a machine learning pipeline to predict the dose-dependent degradation of two different target proteins—ZBTB11 and IKZF1—upon treatment with CRBN-recruiting molecular glue degraders (MGDs). By combining 3D structural features derived from RoseTTAFold 3 (RF3) modeling and OpenMM/OpenFF energy minimization with 2D molecular graph representations, this dual-target neural network provides quantitative predictions of protein abundance across a 10-point concentration gradient.

## Biological Problem and Motivation
Molecular glues are a powerful class of targeted protein degraders, but predicting their selectivity and efficacy remains a massive challenge in drug discovery. While CRBN-based MGDs have historically targeted IKZF1/3, recent studies show they can also recruit novel neosubstrates like ZBTB11. Accurately predicting the degradation profiles for both targets simultaneously is crucial for developing highly selective therapeutics and minimizing off-target toxicity. This workflow bridges 3D structural dynamics with graph neural networks to accurately map glue chemistry to distinct biological outcomes.

## Data Sources & Features Used
The model integrates both 2D chemical data and 3D structural modeling data:

**Chemical Structures:** SMILES strings representing the MGD library.

**Biological Data:** Experimental DMSO-normalized abundance values for ZBTB11 and IKZF1 at 10 test MGD concentrations (%).

**3D Structural Features:** 
* ipTM confidence scores extracted from RoseTTAFold 3 (RF3) modeling of the CRBN:MGD:substrate ternary complexes.
* Minimized complex energies calculated using OpenFF/OpenMM.
* The fractional ratio of successful, physically plausible CRBN:MGD:substrate conformations.

## Computational Approach & Workflow Overview
The primary workflow utilizes a Multi-target Directed Message Passing Neural Network (D-MPNN) implemented using Chemprop v2 and PyTorch Lightning. An Optuna wrapper is used to perform rigorous hyperparameter tuning (optimizing depth, hidden dimensions, dropout, learning rate, temperature, and contrastive loss weights).

To establish a baseline, a Ridge Regression model was trained using Morgan fingerprints combined with the 3D modeling features. Data splitting was strictly controlled using computed Bemis-Murcko scaffolds via the astartes package (80/10/10% train/validation/test split) to ensure the model evaluates true chemical generalization rather than memorization.

## Route Design (Model Architecture)
To handle the dual targets, the model utilizes a bifurcated architecture. It processes the molecular graphs and 3D features together, passes them through a shared predictor, and then splits the network into two highly specialized Feed-Forward Networks (FFNs) for ZBTB11 and IKZF1. The model is trained using a Hybrid MSE Curve Fitting + Contrastive Learning (NT-Xent) Loss Function to better cluster the latent space.
  ```mermaid
graph TD;
    A[SMILES Strings] --> C[Bond Message Passing]
    C --> D[Mean Aggregation]
    B[3D Features: ipTM, Energy, Ratio] --> E
    D --> E[Shared Predictor Regression FFN]
    E --> F[Target Specialization Split]
    
    F --> G[ZBTB11 FFN]
    F --> H[IKZF1 FFN]
    
    G --> I[Predicted ZBTB11 Abundance Array]
    H --> J[Predicted IKZF1 Abundance Array]

    E --> K[Projection Head Sequential]
    K --> L((NT-Xent Contrastive Loss))
    I --> M((MSE Loss))
    J --> M((MSE Loss))
  ```
## Results Summary
**Hyperparameter Optimization:** Optuna successfully evaluated dozens of architectural configurations. The best-performing model configuration (e.g., Trial 11) achieved a significantly minimized validation loss of 0.6385.

**Performance:** The Chemprop D-MPNN model effectively outperformed the Ridge Regression baseline, demonstrating that the bifurcated architecture and contrastive learning loss successfully capture the distinct binding thermodynamics of IKZF1 versus ZBTB11.

(Note to author: Insert links here to any UMAP/loss curve plots or specific R² metrics generated in your EDA).

## Interpretation, Limitations, and Future Directions
**Interpretation:** The bifurcated network design successfully captures the shared mechanics of CRBN recruitment in the early layers, while the specialized branches map the distinct substrate degrons and improve upon a single, unified FFN header. Incorporation of 3D-derived features 

**Limitations:** The network heavily relies on the accuracy of in silico RF3 structures; inaccuracies in generative 3D folding cascade directly into the structural features.

**Future Directions:** Expand the MGD library to test extrapolative capacity on novel IMiD (e.g. thalidomide, lenalidomide) derivatives, and incorporate additional structural featurizations such as Buried Surface Area (BSA) or predicted interface clashes. Other biological parameters could also be featurized, such as half-life of the endogenous proteins, which may improve the power of the model to distinguish drivers of ZBTB11 vs. IKZF1 degradation.

## Reproducibility Instructions
To run this pipeline locally or on Google Colab, ensure you install the required dependencies:

    pip install torch --index-url https://download.pytorch.org/whl/cu[your CUDA version]
    pip install pytorch-metric-learning optuna lightning chemprop ipywidgets umap-learn matplotlib seaborn astartes

1. Clone this repository.

2. Ensure your pre-computed RF3 features and abundance CSVs are placed in the data/ directory.

3. Run model7.ipynb sequentially to execute the data loading and splitting, baseline Ridge Regression, and Chemprop/SupCon model pipeline.
