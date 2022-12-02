import torchplate
from torchplate import experiment
from torchplate import utils
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import requests
import pdb
import cloudpickle as cp
from urllib.request import urlopen
import rsbox 
from rsbox import ml, misc




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*64*64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 50)

    def forward(self, x):
        # grayscale to rgb if needed 
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class IconsExp(torchplate.experiment.Experiment):
    def __init__(self): 
        self.model = Net()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        dataset = ml.classification_dataset("icons-50", resize=(64, 64), normalize=True, extension="png")
        self.trainloader, self.testloader = torchplate.utils.get_xy_loaders(dataset)

        # inherit from torchplate.experiment.Experiment and pass in
        # model, optimizer, and dataloader 
        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            verbose = True
        )
    
    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        x, y = batch
        logits = self.model(x)
        loss_val = self.criterion(logits, y)
        return loss_val

    def test(self):
        accuracy_count = 0
        for x, y in self.testloader:
            logits = self.model(x)
            pred = torch.argmax(F.softmax(logits, dim=1)).item()
            print(f"Prediction: {pred}, True: {y.item()}")
            if pred == y:
                accuracy_count += 1
        print("Accuracy: ", accuracy_count/len(self.testloader))

    def on_epoch_end(self):
        # to illustrate the concept of callbacks 
        print("------------------ (Epoch end) --------------------")



exp = IconsExp()
exp.train(num_epochs=5)
exp.test()
