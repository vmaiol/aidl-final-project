#constantes conf vars
conf_vars = {
    "MAX_NUM_EPOCHS" : 10, #de mom no la utilizo
    "NUM_SAMPLES" : 1, #numero de veces a ejecutar la config/search, especialmente random_search
    "DATA_DIR" : "./../data_sample_victor",
    #"DATA_DIR" : "./data_reduced/",
    "cpu": 2,
    "gpu": 1,
    'config_owner':"maiol",
    'config_type':"mlp",
    'config_search':"other" #grid_search, random_search, y other para tests simples. Valores en hyperparameter_config.py
}
