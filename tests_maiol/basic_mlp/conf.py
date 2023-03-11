#constantes conf vars
maiol_conf_vars = {
    #ray tune
    "MAX_NUM_EPOCHS" : 50, # es un máximo. Si en la connfig ponemos que rayTune escoja entre los valores de la lista, esta variable será el limite
    "NUM_SAMPLES" : 1, #numero de veces a ejecutar la config/search
    "DATA_DIR" : "./../data_sample_victor/",
}
