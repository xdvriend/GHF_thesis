# Installation

## 1. Install and activate virtual environment

The quickest way to install all dependencies is through [Conda](https://docs.conda.io/en/latest/):

```
conda env create -f environment.yml
```

Be sure to activate this environment before starting up your Python interpreter. 

```
source activate ghf
```

## 2. Install the GHF code in the virtual environment

To install the ghf in this in your virtual environment, run:

```
python setup.py install
```

in the ghf folder. 

## 3. Checking the installation and using the library

After the installation please check that all tests run successfully on your system by running following command in the root folder

```
pytest
```

Before using the GHF code, always activate the corresponding virtual environment. In VSCode you can set the Python interpreter explicitly by typing Shift+Cmd+P and typing `Python: Select interpreter`. 
