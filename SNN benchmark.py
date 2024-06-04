# Here to try and time the entire run time of the program
from datetime import datetime
current_time = datetime.now()

hour = current_time.hour
minute = current_time.minute
second = current_time.second
print(f"Hour: {hour}, Minute: {minute}, Second: {second}")

import torch 
import torchvision 
import torchvision.transforms as transforms 
import spikingjelly.clock_driven.encoding as encoding 
import spikingjelly.clock_driven.neuron as neuron 
  

# Hyperparameters 
# Change these to optimise results to balance acuracy and speed
batch_size = 10
n_epochs = 1
input_size = 28 * 28 
num_classes = 10  
lr = 0.01 
  

# Define the SNN model 
class SNN(torch.nn.Module): 
    def __init__(self): 
        super(SNN, self).__init__() 
        # (spike generator) 

        self.snn_input = encoding.PoissonEncoder() 

        self.neuron = neuron.LIFNode() 
 
        self.output_layer = torch.nn.Linear(input_size, num_classes) 
  

    def forward(self, x):        
        x = x.view(x.size(0), -1)                  
        x = self.snn_input(x)         
        x = self.neuron(x) 
        x = self.output_layer(x) 
        return x 

# Load MNIST dataset 
transform = transforms.Compose([ 
    transforms.ToTensor(), 
]) 

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True) 
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) 
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)   

# Initialize the SNN model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
snn_model = SNN().to(device) 

# Define loss function and optimizer 
criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(snn_model.parameters(), lr=lr) 
  

# Train the SNN model 
for epoch in range(n_epochs): 
    running_loss = 0.0 
    for i, (inputs, labels) in enumerate(trainloader, 0): 
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()   
        outputs = snn_model(inputs)     
        # Calculate loss using CrossEntropyLoss 
        loss = criterion(outputs, labels)          
        loss.backward() 
        optimizer.step()   
        running_loss += loss.item() 
      if i % 100 == 99:  # Print every 100 mini-batches 
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100)) 
            running_loss = 0.0   
print('Finished Training')   

# Test the SNN model 
correct = 0 
total = 0 

with torch.no_grad(): 
    for inputs, labels in testloader: 
        inputs, labels = inputs.to(device), labels.to(device) 
        outputs = snn_model(inputs) 
        _, predicted = torch.max(outputs, 1) 
        total += labels.size(0) 
        correct += (predicted == labels).sum().item() 

print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total)) 
End_time = datetime.now()
ahour = End_time.hour
aminute = End_time.minute
asecond = End_time.second
print(f"Hour: {ahour}, Minute: {aminute}, Second: {asecond}")
