import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

data_dir = "./"

test_df = pd.read_csv(data_dir+'sign_mnist_test/sign_mnist_test.csv')
train_df = pd.read_csv(data_dir+'sign_mnist_train/sign_mnist_train.csv')

def dataframe_to_nparray(train_df, test_df):
    train_df1 = train_df.copy(deep = True)
    test_df1 = test_df.copy(deep = True)
    train_images = train_df1.iloc[:, 1:].to_numpy(dtype = 'float32')
    test_images = test_df1.iloc[:, 1:].to_numpy(dtype = 'float32')
    return train_images,test_images

train_img, test_img = dataframe_to_nparray(train_df, test_df)
train_labels = train_df['label'].values
test_labels = test_df['label'].values

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

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ASLBase(nn.Module):
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
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class ASLCNNModel(ASLBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 28, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(28, 28, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     #image size : 28*14*14 

            nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # image size : 56*7*7

            nn.Flatten(), 
            nn.Linear(56*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))
        
    def forward(self, xb):
        return self.network(xb)

model = ASLCNNModel(in_channels, num_classes)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

model.load_state_dict(torch.load('./ModeloEntrenado.pth'))
print(evaluate(model, val_dl))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

def predict_image(img):
    # Convert to a batch of 1
    imgAnp = to_tensor(img)
    xb = to_device(imgAnp.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    values,preds = torch.topk(yb,k = 3,dim = 1)
    # Retrieve the class label
    return preds[0]

'''
counter = 0
for img, label in test_ds:
    if Classes[label.item()] == Classes[predict_image(img, model)]:
        counter += 1

print('Buenas: ',counter,' De: ',len(test_ds), ' Con un error del: ', abs(((counter - len(test_ds))/len(test_ds))*100),'%')
'''
