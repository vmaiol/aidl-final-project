# Define the Accuracy metric in the function below by:
from torchmetrics.functional import r2_score
import torch


def predict(inputs, network):
  network.eval()
  prediction = network(inputs)

  return prediction

def prec_lowres(input):
  mid_position = []
  for data in input:
    mid = data[0].view(-1)
    mid_position.append(mid[20200])
  return torch.tensor(mid_position)

def accuracy(pred_b, tar_b):
  accuracy = abs((pred_b - tar_b))
  accuracy_correction = []
  for acc in accuracy:
    if acc>0.01:
      accuracy_correction.append(1)
    else:
      accuracy_correction.append(0)
  return sum(accuracy_correction)/len(accuracy_correction)

def R2Scores(predicted_batch, target_batch):
  acc=r2_score(predicted_batch, target_batch)
  return acc

def train_epoch(train_loader, network, optimizer, criterion, hparams, epoch):

  # Activate the train=True flag inside the model
  network.train()
  
  device = hparams['device']
  avg_loss = None
  avg_weight = 0.1
  for batch_idx, (data, target) in enumerate(train_loader):
      #print(len(data[0][0].view(-1)))
      low_res = prec_lowres(data)
      data, target = data.to(device), target.unsqueeze(1).to(device)
      optimizer.zero_grad()
      output = network(data)
      loss_data = low_res - output
      print(loss_data)
      loss = criterion(output, target)
      loss.backward()
      if avg_loss:
        avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
      else:
        avg_loss = loss.item()
      optimizer.step()
      if batch_idx % hparams['log_interval'] == 0:
          #print('Batch_idx: {}, Log_interval: {}'.format(batch_idx,hparams['log_interval']))
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))
  return avg_loss



def test_epoch(test_loader, network, criterion, hparams):

    # Dectivate the train=True flag inside the model
    network.eval()
    
    device = hparams['device']
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            # Load data and feed it through the neural network
            data, target = data.to(device), target.unsqueeze(1).to(device)
          

            output = network(data)
            
            # Apply the loss criterion and accumulate the loss
            test_loss += criterion(output, target) # sum up batch loss

            # WARNING: If you are using older Torch versions, the previous call may need to be replaced by
            # test_loss += criterion(output, target, size_average=False).item()

            # compute number of correct predictions in the batch
            acc += R2Scores(output, target)

    # Average accuracy across all correct predictions batches now
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * acc / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc


def fit(model, train_loader, test_loader, hparams, optimizer, criterion ):
    train_losses = []
    test_losses = []
    test_accs = []
    train_loss

    # For each epoch
    for epoch in range(1, hparams['num_epochs'] + 1):

        # Compute & save the average training loss for the current epoch
        train_loss = train_epoch(train_loader, model, optimizer, criterion, hparams)
        train_losses.append(train_loss)

        # TODO: Compute & save the average test loss & accuracy for the current epoch
        # HELP: Review the functions previously defined to implement the train/test epochs
        test_loss, test_accuracy = test_epoch(test_loader, model, hparams)

        test_losses.append(test_loss)
        test_accs.append(test_accuracy)
    return model, test_losses, test_accs