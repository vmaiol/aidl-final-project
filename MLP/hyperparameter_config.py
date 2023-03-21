import numpy as np
import ray #nse si hace falta
from ray import tune
from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
from conf import *

'''Grid search and random search como propuesta de hyperparametr tunning'''
#https://towardsdatascience.com/hyperparameter-tuning-with-grid-search-and-random-search-6e1b5e175144
#tmb explicado un poco en la session 2 o 3 de MLOPS
'''Ray Tune config'''
#https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html?highlight=transformer
#recomendado/propuesto en MLOPS

'''
Principalmente en este archivo definimos la config (hyperparametros a testear en el training),
un reporter (el cual va a guardar las loss o otros valores que querramos analizar despeus del training)
y un numero máximo de num-epochs (en principio deberia tener mas importante con un random search, donde los valores son mas aleatorios seguramente)
'''

def load_hyper_conf(config_owner, config_type, config_search):
    #aqui hago algunos if's por si alguien quiere hacerse sus propias configs!
    ''' ----------- MAIOL ----------------------------------'''
    if config_owner == "maiol":
        if config_type == "mlp": #tipo de modelo
            MAX_NUM_EPOCHS = conf_vars['MAX_NUM_EPOCHS'] #no la utilizo de momento
            #nos ayuda a configurar el tune.report. Cuando acabe de entrenar, le pediremos a rayTune con best_trial = results.get_best_trial("val_loss", "min", "last")
                #cual ha sido la mejor config por esta metric y con que mode (en este caso min, la loss más baja)
            #tune.report(train_loss = loss_stats['train'][-1], val_loss=loss_stats['val'][-1])
            reporter = CLIReporter(
                # parameter_columns=["l1", "l2", "lr", "batch_size"],
                metric_columns=["train_loss", "val_loss", "training_iteration"]) #train loss en principio no la utilizamos

            if config_search =='grid_search': #yo lo he preparado para 2 tipos de de hyper tunn: grid search y random search
                config = {
                    "img_size": tune.grid_search([32, 64, 128]),
                    "img_vars" : tune.choice([1]),
                    "batch_size": tune.grid_search([32, 64, 128]),
                    "n_epochs": tune.choice([10]),
                    "l2": tune.loguniform(1e-5, 1e-1),
                    "lr": tune.loguniform(1e-4, 1e-1),
                }

            elif config_search =='random_search':
                #EN EL RANDOM SEARCH TIENE IMPORTANCIA EL NUM DE SAMPLES!!
                config = {
                    "img_size": tune.choice([28, 32, 64, 128]),
                    "img_vars" : tune.choice([1]),
                    "batch_size": tune.choice([32, 64, 128]),
                    "n_epochs": tune.choice([10]),
                    "l2": tune.loguniform(1e-5, 1e-1),
                    "lr": tune.loguniform(1e-4, 1e-1),
                }
            elif config_search == "other":
                #algo simple para testear o para utilizarlo con la mejor combinacion obtenida en random searc o grid search
                '''simple config para testear...'''
                config = {
                    "img_size": 64,
                    "img_vars" : 1,
                    "batch_size": 32,
                    "n_epochs": 10,
                    #"l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
                    #"l2": tune.loguniform(1e-5, 1),
                    "l2": 0.0015437599875849839,
                    "lr": 0.01898635166558231,
                }

        ''' ----------- FIN CONFIG MAIOL ----------------------------------'''

    elif config_owner == "victor":
        '''VICTOR CONFIG'''
        MAX_NUM_EPOCHS = conf_vars['MAX_NUM_EPOCHS']
        reporter = CLIReporter(metric_columns=["train_loss", "val_loss", "training_iteration"])
        if config_search =='grid_search':
            config = {}
        elif config_search =='random_search':
            config = {}
        elif config_search == "other":
            '''simple config para testear...'''
            config = {}

    elif config_owner == "joan":
        '''JOAN CONFIG'''
        MAX_NUM_EPOCHS = conf_vars['MAX_NUM_EPOCHS']
        reporter = CLIReporter(metric_columns=["train_loss", "val_loss", "training_iteration"])
        if config_search =='grid_search':
            config = {}
        elif config_search =='random_search':
            config = {}
        elif config_search == "other":
            '''simple config para testear...'''
            config = {}

    return MAX_NUM_EPOCHS, reporter, config
