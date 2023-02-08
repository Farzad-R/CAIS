# Carleton AI Society

The repository is originated from the following paper:

H. Zheng, F. Lin, X. Feng and Y. Chen. [A Hybrid Deep Learning Model with Attention based Conv-LSTM Networks for Short-Term Traffic Flow Prediction](https://ieeexplore.ieee.org/document/9112272). IEEE Transactions on Intelligent Transportation Systems, Preprint, 2020.

With the following [Github repository](https://github.com/suprobe/AT-Conv-LSTM).


Prerequisites:
```
- tensorflow-gpu            2.6.0 
- keras                     2.6.0
- numpy                     1.23.5
```

Installation (using conda):
- Open the terminal and enter the directory that you would like to have the repository.
- Run:
```
git clone https://github.com/Farzad-R/CAIS.git
conda create --name tf_gpu_cais tensorflow-gpu==2.6.0
activate tf_gpu_cais
pip install numpy==1.23.5
python Train.py
```
