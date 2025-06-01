# Simplified PyTorch Implementation of Improved AE for Unsupervised Anomaly Detection

This [PyTorch](https://pytorch.org/) implementation of the *Improved AE for Unsupervised Anomaly Detection* (IAEAD) method is a simplified version based on the paper by Cheng et al. (doi: 10.1002/int.22582). This implementation is a modification of a simplified Deep SVDD implementation according to the details in the Cheng et al. paper. The original IAEAD implementation is available at [pt-iaead](https://github.com/wogong/pt-iaead).

## Overview

Improved AE for Unsupervised Anomaly Detection (IAEAD) is an unsupervised anomaly detection algorithm that combines the strengths of autoencoders and Deep SVDD. The algorithm uses a modified autoencoder architecture with a hypersphere constraint in the latent space to better detect anomalies.

The project includes:
- Implementation of the IAEAD method based on the Cheng et al. (2021) paper.
- Modified autoencoder architecture with hypersphere constraint
- Visualization tools for both latent space and data space
- Evaluation metrics for anomaly detection performance

## Algorithm Description

The IAEAD algorithm combines autoencoder reconstruction with a hypersphere constraint in the latent space:

1. **Training Phase**: The autoencoder is trained with a combined loss function that includes:
   - Reconstruction loss: Measures how well the autoencoder can reconstruct the input data
   - Hypersphere constraint: Encourages the latent representations to be close to a predefined center in the latent space

2. **Anomaly Detection Phase**: Anomalies are detected based on a combined score that considers both:
   - Reconstruction error: How well the input can be reconstructed
   - Distance from center: How far the latent representation is from the center of the hypersphere

The objective function for IAEAD combines these components to create a more robust anomaly detection method that leverages both the reconstruction capabilities of autoencoders and the compact representation of Deep SVDD.

## Requirements
- Python 3.12
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Scikit-learn >= 1.2.0
- tqdm >= 4.65.0
- SciPy >= 1.10.0

All dependencies are listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rjsaltos1989/IAEAD.git
   cd IAEAD
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n iaead python=3.12
   conda activate iaead
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `main.py`: Main script to run the IAEAD algorithm
- `nn_models.py`: Neural network model definitions (AutoEncoder and IAEAD models)
- `nn_train_functions.py`: Functions for training the autoencoder
- `svdd_nn_train_functions.py`: Functions for training the IAEAD model
- `svdd_eval_functions.py`: Functions for evaluating the IAEAD model
- `plot_functions.py`: Functions for visualizing results

## Usage

### Basic Usage

1. Modify the dataset path in `main.py` to point to your data:
   ```python
   dataset_path = '/path/to/your/data/'
   dataset_file = 'YourDataset.mat'
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

### Customization

You can customize the following parameters in `main.py`:

- `latent_dim`: Dimension of the latent space (default: 2)
- `nu`: Hyperparameter controlling the fraction of outliers (default: 0.03)
- `train_epochs`: Number of epochs for IAEAD training (default: 100)
- `batch_size`: Batch size for training (default: 32)
- `gamma`: Weight parameter balancing reconstruction loss and hypersphere constraint

### Input Data Format

The code expects data in MATLAB .mat format with:
- 'Data': Matrix where rows are samples and columns are features
- 'y': Vector of labels where anomalies are labeled as 2

## Example Results

When running the code, you'll get:
1. Training loss plots showing the convergence of the model, including both reconstruction loss and hypersphere constraint components
2. Visualization of the data in the latent space, showing:
   - Normal data points
   - Anomalous data points
   - The center of the hypersphere
3. Performance metrics including AUC-ROC, AUC-PR, F1-Score, and Recall

## References

- IAEAD Paper: Cheng, M., Xu, Q., Lv, J., Liu, W., Li, Q., & Wang, J. (2021). Improved autoencoder for unsupervised anomaly detection. International Journal of Intelligent Systems, 36(11), 6436-6459. doi: 10.1002/int.22582
- Original IAEAD Implementation: [wogong/pt-iaead](https://github.com/wogong/pt-iaead)
- Deep SVDD Paper: Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Müller, E., & Kloft, M. (2018). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
