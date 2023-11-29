
# EP02: Transformers Regression and Quantization Project

## Overview
This project involves a detailed exploration of refining encoders in the Transformers architecture, specifically focusing on BERT-like models. The primary goal is to assess the performance of these models in numerical regression tasks and explore solutions for improving their performance in such scenarios. The project uses the B2W review corpus, focusing on the 'review text' column, and involves tasks related to vowel density regression and quantization.

## Installation
To set up the environment for running this notebook, please ensure Python and the necessary libraries are installed. You can install all required dependencies via the following command:
```
pip install -r requirements.txt
```

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Libraries: Pandas, NumPy, PyTorch, Transformers (Hugging Face)

## Usage
To use this notebook, navigate to the directory containing `EP02_main.ipynb` and run:
```
jupyter notebook EP02_main.ipynb
```
Follow the instructions within the notebook to execute the cells.

## Notebook Content
The `EP02_main.ipynb` notebook contains several key components:
- **GPU and RAM Setup**: Checks for GPU availability and RAM size to ensure high-performance computation.
- **Drive Mounting**: Code to mount Google Drive for accessing data files.
- **Preprocessing**: Scripts for data preprocessing, including handling of the B2W review corpus.
- **Baseline Models**: Establishment of baseline models for comparison in regression tasks.
- **Training Regression Models**: Training of BERT models for vowel density regression.
- **Testing Regression Models**: Evaluation of the trained regression models.
- **Training and Testing Classification Models**: Implementation of quantization tasks for vowel density using classification models.

## System Requirements
- A system with Python 3.x installed.
- Access to a GPU is recommended for faster computation, but the notebook can also run on a CPU.
- Sufficient RAM (at least 8GB) for handling data preprocessing and model training.

## Contribution
Contributions to this project are welcome. Please adhere to the project's coding standards and submit pull requests for any enhancements.

## License
This project is released under the MIT License.

## Contact
For any queries or further discussion related to this project, feel free to contact the maintainers.
