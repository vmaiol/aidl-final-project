#constantes conf vars
conf_vars = {
    #ray tune
    "MAX_NUM_EPOCHS" : 10, # es un máximo. Si en la connfig ponemos que rayTune escoja entre los valores de la lista, esta variable será el limite
    "NUM_SAMPLES" : 1, #numero de veces a ejecutar la config/search
    "DATA_DIR" : "./../data_sample_victor",
    #"DATA_DIR" : "/home/juanrivas/data_reduced/"
    "cpu": 2,
    "gpu": 1,
    'config_owner':"maiol",
    'config_type':"mlp",
    'config_search':"other" #grid_search, random_search, y other para tests simples
}
