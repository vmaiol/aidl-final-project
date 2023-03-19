
import torch
import matplotlib.pyplot as plt


def predict(inputs, network):
  network.eval()
  prediction = network(inputs)

  return prediction

def prec_lowres(input):
  """
  Function to extract the low resolution precipitation of the input data.
  """
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

def train_epoch(train_loader, network, optimizer, criterion, hparams, epoch):
  """
  Function to save the loss and accuracy plots to disk.
  """
  # Activate the train=True flag inside the model
  network.train()
  
  device = hparams['device']
  avg_loss = None
  avg_weight = 0.1
  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.unsqueeze(1).to(device)
      optimizer.zero_grad()
      output = network(data)
      loss = criterion(output, target)
      loss.backward()
      if avg_loss:
        avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
      else:
        avg_loss = loss.item()
      optimizer.step()
      if batch_idx % hparams['log_interval'] == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))
  return avg_loss



def val_epoch(val_loader, network, criterion, hparams):
  """
  Function to validate the training
  """

  # Dectivate the train=True flag inside the model
  network.eval()
    
  device = hparams['device']
  val_loss = 0
  val_losses = []
  with torch.no_grad():
    for data, target in val_loader:
            
      # Load data and feed it through the neural network
      data, target = data.to(device), target.unsqueeze(1).to(device)

      output = network(data)
            
      # Apply the loss criterion and accumulate the loss
      val_loss = criterion(output, target) # sum up batch loss
      val_losses.append(val_loss)

      # Average loss across all batches
    print('\nValidation set: Average loss: {:.4f}\n'.format(
      sum(val_losses)/len(val_losses)
      ))
  return sum(val_losses)/len(val_losses)


def fit(model, train_loader, val_loader, hparams, optimizer, criterion):
  """
  Function to train the model.
  """
  train_losses = []
  val_losses = []
  train_loss= []
  val_loss= []

  # For each epoch
  for epoch in range(1, hparams['num_epochs'] + 1):
    # Compute & save the average training loss for the current epoch
    train_loss = train_epoch(train_loader, model, optimizer, criterion, hparams, epoch)
    train_losses.append(train_loss)

    
    val_loss = val_epoch(val_loader, model, criterion, hparams)

    val_losses.append(val_loss)
  return model, train_losses, val_losses


def save_plots(train_loss, valid_loss):
  """
  Function to save the loss and accuracy plots to disk.
  """
  # loss plots
  plt.figure(figsize=(10, 7))
  plt.plot(
    train_loss, color='orange', linestyle='-', 
    label='train loss'
  )
  plt.plot(
    valid_loss, color='red', linestyle='-', 
    label='validation loss'
  )
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('outputs/loss.png')
  plt.show()


def save_model(epochs, model, optimizer, criterion):
  """
  Function to save the trained model to disk.
  """
  print(f"Saving final model...")
  torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
    }, 'outputs/final_model.pth')