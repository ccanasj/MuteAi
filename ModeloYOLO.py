import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from PIL import Image
import pandas as pd

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from torchvision.utils import make_grid

data_dir = "./"

test_df = pd.read_csv(data_dir+'sign_mnist_test/sign_mnist_test.csv')
train_df = pd.read_csv(data_dir+'sign_mnist_train/sign_mnist_train.csv')

Classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def dataframe_to_nparray(train_df, test_df):
    train_df1 = train_df.copy(deep = True)
    test_df1 = test_df.copy(deep = True)
    train_images = train_df1.iloc[:, 1:].to_numpy(dtype = 'float32')
    test_images = test_df1.iloc[:, 1:].to_numpy(dtype = 'float32')
    return train_images,test_images

train_img, test_img = dataframe_to_nparray(train_df, test_df)
train_labels = train_df['label'].values
test_labels = test_df['label'].values

print(train_img.shape[0])
train_images_shaped = train_img.reshape(train_img.shape[0],1,28,28)
test_images_shaped = test_img.reshape(test_img.shape[0],1,28,28)

train_images_tensors = torch.from_numpy(train_images_shaped)
train_labels_tensors = torch.from_numpy(train_labels)

test_images_tensors = torch.from_numpy(test_images_shaped)
test_labels_tensors = torch.from_numpy(test_labels)

train_ds_full = TensorDataset(train_images_tensors, train_labels_tensors)
test_ds = TensorDataset(test_images_tensors, test_labels_tensors)

# Hyperparmeters
batch_size = 64
learning_rate = 0.001

# Other constants
in_channels = 1
input_size = in_channels * 28 * 28
num_classes = 26

random_seed = 11
torch.manual_seed(random_seed);

val_size = 7455
train_size = len(train_ds_full) - val_size

train_ds, val_ds = random_split(train_ds_full, [train_size, val_size,])

train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size*2, pin_memory=True)

class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, in_channels*28*28)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = ASLModel()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, in_channels*28*28)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = ASLModel()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
  
for images, labels in test_dl:
    outputs = model(images)
    print(labels)
    print(accuracy(outputs, labels))
    break

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

result = evaluate(model, test_dl)
print(result)

history1 = fit(10, 0.001, model, train_dl, val_dl)

result = evaluate(model, test_dl)
print(result)

history2 = fit(10, 0.0001, model, train_dl, val_dl)

result = evaluate(model, test_dl)
print(result)

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

counter = 0
for img, label in test_ds:
    if Classes[label.item()] == Classes[predict_image(img, model)]:
        counter += 1



print('Buenas: ',counter,' De: ',len(test_ds), ' Con un error del: ', abs(((counter - len(test_ds))/len(test_ds))*100),'%')

a = Image.open("./a.jpg").convert('L')
h = Image.open("./h.jpg").convert('L')
imagenA = a.resize((28,28))
imagenH = h.resize((28,28))

imgA = to_tensor(imagenA)
imgH = to_tensor(imagenH)
print(Classes[predict_image(imgA, model)])

print(imgH.shape)
imgH = torch.flatten(imgH)
print(Classes[predict_image(imgH, model)])
