### To execute
```
conda create --name aidl-final-project-mlp python=3.8
```
To activate the environment:
```
conda activate aidl-final-project-mlp
```
and install the dependencies
```
pip install -r requirements.txt
```

To stop/deactivate the environment:
```
conda deactivate
```

Execute basic structure to check that minimums have been installed. In the project folder:
```
python3 main_regression.py
```

Outputs and progress inside outs_mlp_train. Each folder is a trial (combination of hyperparameters) and saves some results/progress. In addition the model and the plot of train and val losses.

Test plot will be in root folder and will be tested with the best config of the train.

To test different configs:
- Edit conf.py
