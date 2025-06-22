🚀 Project Overview
This repository provides a complete workflow for using 1D Convolutional Autoencoders to perform unsupervised clustering of time-series signals. The core idea is to find an optimal model that not only reconstructs signals accurately but also creates a meaningful, well-separated latent space for distinguishing different signal classes.

The project leverages a multi-objective Pareto optimization to balance two critical goals:

📉 Low Reconstruction Error: The model's ability to compress and then accurately reconstruct input signals.

📈 High Clustering Quality: The model's ability to group similar signals together in its latent space.

The process is broken down into three automated phases: Training, Evaluation, and Testing.

📂 File Structure
Here's a breakdown of the key files in this project:

File Name

Description

data_and_train.m

🧠 Main Training Script: Trains 40 different autoencoder models using a hyperparameter grid search. Generates clustering_models_SNR_inc.mat.

test_phase1.m

📊 Evaluation & Selection Script: Evaluates all trained models using clustering metrics and Pareto analysis to select the single best model.

test2.m

🔬 Performance Testing Script: Rigorously tests the best model's classification and rejection performance on signals with varying levels of noise (SNR).

daviesBouldin.m

🛠️ Utility Function: Calculates the Davies-Bouldin Index to evaluate clustering quality.

combined_1000.mat

📦 Input Data: The initial dataset of signals (not included).

clustering_models...mat

💾 Generated Models: Contains all trained networks, created by data_and_train.m.

test_data.mat

🧪 Generated Test Data: Contains noisy test signals, created by test2.m.

⚙️ Methodology Workflow
The project follows a sequential pipeline where each script's output is the next one's input.

Phase 1: Model Training (data_and_train.m)
In this phase, we perform a wide search to create a diverse set of candidate models.

Data Preparation: Loads and normalizes the signals from combined_1000.mat.

Hyperparameter Grid: Defines a search space for latent dimensions, filter sizes, and data augmentation levels.

Automated Training: Iteratively trains 40 unique 1D convolutional autoencoders.

Model Saving: Exports all trained networks, encoders, and training metadata to clustering_models_SNR_inc.mat.

Phase 2: Evaluation & Selection (test_phase1.m)
Here, we analyze the trade-offs and intelligently select the most balanced and effective model.

Metric Calculation: For each of the 40 models, it computes the Silhouette Score and Davies-Bouldin Index.

Pareto Front Analysis: It plots the trade-off between Reconstruction Error (lower is better) and a Unified Cluster Score (higher is better). This visualization helps identify the set of non-dominated, optimal models.

Best Model Selection: The script automatically picks the best compromise model from the Pareto front—the one closest to the ideal "Utopian Point" (zero error, perfect score).

Phase 3: Performance Testing (test2.m)
The final phase subjects our chosen model to a rigorous stress test to see how it performs in a more realistic, noisy environment.

Noisy Data Generation: Creates a large test set by adding varying levels of Gaussian noise to the signals.

Robustness Evaluation: The best model classifies the noisy signals and uses a reconstruction error threshold to reject unrecognizable or anomalous inputs.

Performance Metrics: It computes detailed metrics like the Macro F1-score and confusion matrices to quantify classification accuracy at different noise levels.

Advanced Visualization: Generates and saves insightful figures in the performance_results/ folder, including 3D error surfaces and confusion charts.

🚀 How to Run
Prerequisites
MATLAB (R2023b or newer recommended)

Deep Learning Toolbox

Statistics and Machine Learning Toolbox

The raw signal data file combined_1000.mat must be available in the MATLAB path.

Execution
Execute the scripts from the MATLAB command window or editor in the following order:

    % 1. Train all 40 autoencoder models. This may take a significant amount of time.
    >> run('data_and_train.m');

    % 2. Evaluate the models and visualize the Pareto front to select the best one.
    >> run('test_phase1.m');

    % 3. Generate the noisy test set and evaluate the final model's performance.
    >> run('test2.m');

📊 Outputs
The scripts will automatically generate and populate the following directories with figures and data files:

reconstruction_quality_figures/: Plots showing reconstruction quality for each of the 40 trained models.

performance_results/: Contains the final performance plots for the best model, including 3D scatter plots and confusion matrices.

Happy coding!
