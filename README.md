# $D^2FL$ 

## Overview

This repository provides the code and instructions for reproducing the experiments presented in the paper  *$D^2FL$: Dimensional Disaster-oriented Backdoor Attack Defense Of Federated Learning*. The primary goal of this project is to reproduce results.

## Environment Setup

To reproduce the experiments, please ensure your system meets the following requirements: 

- **Operating System**: Ubuntu 20.04.5 LTS
- **Python version**: Python 3.8
-  **Libraries**: Please see `requirements.sh` for a complete list of required packages. 
- **Hardware**: NVIDIA GeForce RTX 4090D with CUDA 11.8

### Additional Requirements 

- **Dataset**: Datasets are automatically downloaded.
- **Pre-trained Models**: If using pre-trained models, see the [Pretrained Models](#pretrained-models) section.

## Installation

1.  Clone this repository:
    ```sh
    git clone https://github.com/Elonisme/D2FL.git
    ```

2. Install required dependencies:
    ```sh
    chmod +x ./requirements.sh
    ./requirements.sh
    ```

## Training and Testing
### Quick Start
```sh
python main.py
```

### Detailed Instructions
By setting the mode to mul, single, or queue, you can determine whether the program operates in multiprocessing, single-process, or queuing mode.
```python
if __name__ == '__main__':
    mode = 'mul'
    if mode == 'mul':
        mul()
    elif mode == 'single':
        single()
    elif mode == 'queue':
        queue()
    else:
        raise KeyError("mode must be either 'queue' or 'mul'")
```

Modify the basic configuration in the get_base_config function.
```python
def get_base_config():
    base_config = {'num_clients': 100, 'client_frac': 0.15, 'malicious_rate': 0.2,
                   'model_name': 'resnet18', 'data_name': 'cifar10', 'aggregate_type': 'flame',
                   'poison_type': 'trigger', 'poisoning_threshold': 5, 'num_epochs': 50,
                   'save_slogan': True, 'fl_print': True, 'sampling_stride': 2, 'alpha': 0.5,
                   'poison_probability': 0.3, 'pretrained': False}
    return base_config
```


## Pretrained Models
If you would like to use pretrained models, download them from the following links:
- [Google Drive](https://drive.google.com/drive/folders/10CkBB68cRyZNjqUdrNnXUHaV8UznF0P9?usp=drive_link)
- Place the downloaded models in the pretrain_weights/ directory 

## References

If you find this repository useful, please consider citing the following paper:

``` bibtex
@article{your_paper_reference,
  title={Paper Title},
  author={Author Name(s)},
  journal={Journal/Conference Name},
  year={Year}
}
```

