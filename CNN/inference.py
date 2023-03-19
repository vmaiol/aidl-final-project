from model import *
from dataset import *
from resources import *

hparams = {
    'batch_size':10,
    'num_epochs':20,
    'test_batch_size':10,
    'hidden_size':128,
    'num_classes':1,
    'learning_rate':1e-4,
    'log_interval':2,
    'early:stop': True
}

hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'


path="models.pt"
model = Downscaling_model1().to(hparams['device'])
model.load_state_dict(torch.load(path))
model.eval()

path_data = "data_input"
files = os.listdir(path_data)
dataset = Downscaling_dataset(path_data)
train, test, val = torch.utils.data.random_split(dataset, [0.7,0.2,0.1])

data_loader = test_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
train_loader = test_loader = torch.utils.data.DataLoader(train, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
test_loader = test_loader = torch.utils.data.DataLoader(test, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
val_loader = test_loader = torch.utils.data.DataLoader(val, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)


for data, target in test_loader:
    data, target = data.to(hparams['device']),target.unsqueeze(1).to(hparams['device'])
    output = model(data)
    print("Prediction:",output)
    print("Target:",target)
    print("Accuracy:",output-target)
    print("Low Resolution target",prec_lowres(data))
    break
