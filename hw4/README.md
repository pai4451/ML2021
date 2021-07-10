# My Environment
* Ubuntu 18.04.5 LTS
* Python 3.7.6
* torch 1.7.0
* cuda 11.0
* cudnn 8.0.3

# Quick  start to run my code

* `git clone https://github.com/Masao-Someki/Conformer.git`
* `export PYTHONPATH="$PWD/Conformer"`
* `python ./hw4.py`

Note that to run the above code `torch` package must be installed. If not, you can use the following commands that will create a virtual environment named as `venv` and the commands will install torch/torchvision and some required dependencies. If you aleady have torch/torchvision in your computer, then these commands are not necessary.

* (Optional) `cd Conformer/tools`
* (Optional) `make` 


# References
* Conformer implementation with Pytorch: 
The conformer I used is implemented by Masao Someki
[https://github.com/Masao-Someki/Conformer]
* Conformer original paper [https://arxiv.org/abs/2005.08100]
* The sample code from TA, NTU machine learning course
