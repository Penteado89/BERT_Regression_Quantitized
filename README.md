
# EP02: Transformers Regression and Quantization Project

## Overview
In this research, we explored the adaptability of Transformer-based models, with a focus on BERT (Bidirectional Encoder Representations from Transformers), for tasks beyond their conventional classification applications. The study harnessed the B2W review corpus, analyzing the 'review text’ variable to compute and categorize the proportion of vowels in the text entries. This approach allowed us to investigate how Transformer architectures, typically employed for classification, can be effectively adapted for numerical regression tasks. Our methodology entailed fine-tuning models such as BertForVowelDensityRegression, BertForQuantizedClassification, and BertForBalancedClassification, all derived from the BERTimbau pre-trained model. This enabled the detailed examination of these models' capabilities in handling tasks like vowel density regression and sentence categorization based on quantized vowel density.

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Libraries: Pandas, NumPy, PyTorch, Transformers (Hugging Face)

## Usage
To use this notebook, navigate to the directory containing `EP02_main.ipynb` and follow the instructions within the notebook to execute the cells.

## Notebook Content
The `EP02_main.ipynb` notebook contains several key components:
- **GPU and RAM Setup**: Checks for GPU availability and RAM size to ensure high-performance computation.
- **Drive Mounting**: Code to mount Google Drive for accessing data files.
- **Preprocessing**: Scripts for data preprocessing, including handling of the B2W review corpus.
- **Baseline Models**: Establishment of baseline models for comparison in regression tasks.
- **Training Regression Models**: Training of BERT models for vowel density regression.
- **Testing Regression Models**: Evaluation of the trained regression models.
- **Training and Testing Classification Models**: Implementation of quantization tasks for vowel density using classification models.
- **Comparing Task 2 and Task 3**
  
## System Requirements
- A system with Python 3.x installed.
- Access to a GPU is recommended for faster computation, but the notebook can also run on a CPU.
- Sufficient RAM (at least 8GB) for handling data preprocessing and model training.

## Contribution
Contributions to this project are welcome. Please adhere to the project's coding standards and submit pull requests for any enhancements.

## Contact
For any queries or further discussion related to this project, feel free to contact the maintainers.
