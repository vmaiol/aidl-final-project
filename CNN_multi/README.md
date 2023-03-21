# Training of the multivariable model
With this experiment you can train the multivariable model with a reduced dataset. The results and the model will be saved in the outputs folder.

## Environment Installation
### With Conda
Creating a conda environment:
```
conda create --name aidl-final-project python=3.8
```
To activate the environment:
```
conda activate aidl-final-project
```
and install the dependencies
```
pip install -r requirements.txt
```
To execute the CNN model with only precipitation variable go to the CNN directory
```
cd CNN_multi
```
And then execute the main.py:
```
python3 main.py
```
