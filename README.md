# Transformer

## Project Description
This project implements a Transformer model for translating text from English to Italian. The model is built using PyTorch and leverages the Huggingface `datasets` and `tokenizers` libraries.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Configuration
You can customize the training configuration by modifying the `config.py` file or by passing parameters to the `get_config` function. Here are the available parameters:

- `batch_size`: Batch size for training (default: 16)
- `num_epochs`: Number of epochs for training (default: 10)
- `lr`: Learning rate (default: 0.0001)
- `seq_len`: Sequence length (default: 350)
- `d_model`: Dimension of the model (default: 512)
- `datasource`: Data source for training (default: 'opus_books')
- `lang_src`: Source language (default: 'en')
- `lang_tgt`: Target language (default: 'it')
- `model_folder`: Folder to save model weights (default: 'weights')
- `model_basename`: Base name for model weights files (default: 'tmodel_')
- `preload`: Preload model weights ('latest' or specific epoch, default: 'latest')
- `tokenizer_file`: Tokenizer file pattern (default: 'tokenizer_{0}.json')
- `experiment_name`: Experiment name for TensorBoard (default: 'runs/tmodel')

Example of getting a custom configuration:
```python
from config import get_config
config = get_config(batch_size=32, num_epochs=20, lr=0.0005)
```

## Usage
### Training the Model
To train the model, run:
```bash
python train.py
```

### Translating Text
To translate a sentence, run:
```bash
python translate.py "Your sentence here"
```

## Running in Google Colab
You can run this project in Google Colab by following these steps:

1. Open the Colab notebook: [English_To_Italian_Using_Transformer_Architecture.ipynb](English_To_Italian_Using_Transformer_Architecture.ipynb)
2. Clone the repository:
    ```python
    !git clone [https://github.com/vinay-852/Transformer.git](https://github.com/vinay-852/Translator_Using_Transformer_Architecture.git)
    ```
3. Install the dependencies:
    ```python
    !pip install -r Transformer/requirements.txt
    ```
4. Change the directory:
    ```python
    %cd /content/Transformer/
    ```
5. Train the model:
    ```python
    from config import get_config
    from train import train_model
    cfg = get_config()
    train_model(cfg)
    ```
6. Translate text:
    ```python
    from translate import translate
    t = translate("Why do I need to translate this?")
    print(t)
    ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
