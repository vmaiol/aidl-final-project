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
            #nos ayuda a configurar el tune.report. Cuando acabe de entrenar, le pediremos a rayTune con best_trial = results.get_best_trial("val_loss", "min", "last")
                #cual ha sido la mejor config por esta metric y con que mode (en este caso min, la loss más baja)
            MAX_NUM_EPOCHS = maiol_conf_vars['MAX_NUM_EPOCHS']
            #para hacer el report cuando acabe de entrenar. Haremos algo parecido a esto en train o val. Segurament en val, ya que si queremos registrar la loss del val, tenemos que esperar a que termine el train_epoch, y val_epoch
            #tune.report(train_loss = loss_stats['train'][-1], val_loss=loss_stats['val'][-1])
            reporter = CLIReporter(
                # parameter_columns=["l1", "l2", "lr", "batch_size"],
                metric_columns=["train_loss", "val_loss", "training_iteration"]) #train loss en principio no la utilizamos

            if config_search =='grid_search': #yo lo he preparado para 2 tipos de de hyper tunn: grid search y random search
                config = {
                    "img_size": tune.grid_search([16, 32, 64, 128]),
                    "img_vars" : tune.choice([1]),
                    "train_batch_size": tune.grid_search([64, 128]),
                    "val_batch_size": tune.grid_search([64, 128]),
                    "n_epochs": tune.choice([50]),
                    #"l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
                    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)), #weight_decay
                    "lr": tune.loguniform(1e-4, 1e-1),
                }

            elif config_search =='random_search':
                #EN EL RANDOM SEARCH TIENE VALOR EL NUM DE SAMPLES!!
                config = {
                    "img_size": tune.grid_search([64]),
                    "img_vars" : tune.choice([1]),
                    "train_batch_size": tune.choice([16, 32, 64]),
                    "val_batch_size": tune.choice([64]),
                    "n_epochs": tune.choice([20]),
                    #"l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
                    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
                    "lr": tune.loguniform(1e-4, 1e-1),
                }
            elif config_search == "other":
                #TODO algo simple para testear?
                '''simple config para testear...'''
                config = {
                    "img_size": tune.grid_search([64]),
                    "img_vars" : tune.choice([1]),
                    "train_batch_size": tune.choice([64]),
                    "val_batch_size": tune.choice([64]),
                    "n_epochs": tune.choice([50]),
                    #"l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
                    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
                    "lr": tune.loguniform(1e-4, 1e-1),
                }

        ''' ----------- FIN CONFIG MAIOL ----------------------------------'''

    elif config_owner == "victor":
        '''VICTOR CONFIG'''

    elif config_owner == "joan":
        '''JOAN CONFIG'''

    return MAX_NUM_EPOCHS, reporter, config
