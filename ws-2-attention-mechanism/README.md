# ATT-Conv-LSTM

The repository is originated from the following paper:

H. Zheng, F. Lin, X. Feng and Y. Chen. [A Hybrid Deep Learning Model with Attention based Conv-LSTM Networks for Short-Term Traffic Flow Prediction](https://ieeexplore.ieee.org/document/9112272). IEEE Transactions on Intelligent Transportation Systems, Preprint, 2020.

With the following [Github repository](https://github.com/suprobe/AT-Conv-LSTM).


## Prerequisites:
Tensorflow gpu has some system requirements of its own and in case you haven't worked with it before, it takes few steps before it can be installed. A detailed explanation of those steps is given [here](https://www.tensorflow.org/install/pip). This project requires the following libraries
```
- tensorflow-gpu            2.6.0 
- keras                     2.6.0
- numpy                     1.23.5
```

## Installation (using conda):
```
cd to the project's repository
conda create --name tf_gpu_cais tensorflow-gpu==2.6.0
activate tf_gpu_cais
conda install -c conda-forge keras
pip install numpy==1.23.5
python Train.py
```

## Project structure:
```
├── data                    # (added to .gitignore)
│   ├── freeway
│   └── urban               # to be filled if you would like to train the model on urban data. Data can be found [here](https://github.com/suprobe/AT-Conv-LSTM)
├── model                   # For saving the model checkpoints 
│ 
├── workshop                # contains some images used in the workshop presentation
│ 
├── Train.py                # the executable module (calling the following modules in the given order)
│       ├── ReadData.py        
│       │       └── DataPreparation.py
│       │
│       ├── utils.py        # contains the loss functions
│       │
│       └── Models.py       # contains two models (with/without attention mechanism)
│               └── AttentionLayer.py
│      
├──AttentionTutorial.ipynb  # the code used to investigate the attention mechanism during the workshop
│
├──XplrAttention.py         # Extra module for further exploration of the attention mechanism used for traffic forecasting
│
└── README.md
```