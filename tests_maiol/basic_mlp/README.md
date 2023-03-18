### Instructions to run the code
```
conda create --name aidl-final-project-mlp python=3.8
```
To activate the environment:
```
conda activate aidl-final-project-mlp
```
Install the dependencies:
```
pip install -r requirements.txt
```

To stop/deactivate the environment:
```
conda deactivate
```

Edit conf.py for "custom" config.
Data directory should be changed, as well as the cpu, gpu, the config search (random search, grid search, or other), or the config owner if some new config has been added.
Grid search and random search have been implemented for hyperparameter tuning. The "other" option is thought in order to test more fast the config or, in additon, for setting the best combination obtained from the hyperparamter tuning tests.

Execute main_regression.py to run the code:
```
python3 main_regression.py
```

Since this code shares an hyperparamter tuning and, perhaps, a "normal" training with defined parameters, outputs and progress/results are saved inside outs_mlp_train because of Ray Tune. Each folder is a trial (combination of hyperparameters to be tested with training etc) and stores some results/progress. In addition to the model and the plot of train and val losses.

Train and val plot will be saved in the trail folder called outs_mlp_train.
Test plot will be in the root folder and will be tested with the best config of the train. Obviously, if we run "other" or just a single combinations of hyperparameters, best config is the same by default...

If we do the hyperparamter config, the final best config obtained is added in the file best_config.txt

Load_data.py can be deleted, it is not used.
