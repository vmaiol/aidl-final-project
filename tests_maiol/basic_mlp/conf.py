#constantes conf vars
conf_vars = {
    "MAX_NUM_EPOCHS" : 10,
    "NUM_SAMPLES" : 1, #numero de veces a ejecutar la config/search
    "DATA_DIR" : "./../data_sample_victor",
    #"DATA_DIR" : "/home/juanrivas/data_reduced/"
    "cpu": 2,
    "gpu": 1,
    'config_owner':"maiol",
    'config_type':"mlp",
    'config_search':"grid_search" #grid_search, random_search, y other para tests simples. Valores en hyperparameter_config.py
}
