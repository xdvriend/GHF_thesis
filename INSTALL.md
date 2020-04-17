# Installation

## Install virtual environment

The quickest way to install all dependencies is through conda:

```
conda env create -f environment.yml
```

Be sure to activate this environment before starting up your Python interpreter. 

```
source activate ghf
```

In VSCode you can set the Python interpreter by typing Shift+Cmd+P and typing `Python: Select interpreter`. After the installation please check that all tests run successfully on your system by running following command in the root folder

```
pytest
```
