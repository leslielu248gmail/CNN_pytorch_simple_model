import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class simpleModel(nn.Model):
  def __init__(self):
    super(simpleModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 5)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, 5)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(32*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x)) # input(3, 32, 32) output(16, 28, 28)
    x = self.pool1(x)         # output(16, 14, 14)
    x = F.relu(self.conv2(x)) # output(32, 10, 10)
    x = self.pool2(x)         # output(32, 5, 5)
    x = x.view(-1, 32*5*5)    # output(32*5*5)
    x = F.relu(self.fc1(x))   # output(120)
    x = F.relu(self.fc2(x))   # output(84)
    x = self.fc3(x)           # output(10)


def train():
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=False, transform=transform)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                             shuffle=True, num_workers=0)

  val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=False, transform=transform)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                           shuffle=False, num_workers=0)

  val_data_iter = iter(val_loader)
  val_image, val_label = next(val_data_iter)

  model = simpleModel()
  loss_function = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(5):

    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
      inputs, labels = data

      optimizer.zero_grad()
      
      outputs = model(inputs)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if step % 500 == 499:
        with torch.no_grad():
          outputs = model(val_image)
          predict_y = torch.max(outputs, dim=1)[1]
          accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

          print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' % 
                (epoch+1, step+1, running_loss/500, accuracy))
          running_loss = 0.0

      print('Finished Training')

      save_path = './simpleModel.pth'
      torch.save(model.state_dict(), save_path)


def predict():
  transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
  )

  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  model = simpleModel()
  model.load_state_dict(torch.load('simpleModel.pth'))

  im = Image.open('1.jpg')
  im = transform(im)
  im = torch.unsqueeze(im, dim=0)

  with torch.no_grad():
    outputs = simpleModel(im))
    predict = torch.max(outputs, dim=1)[1].numpy()
  print(classes[int(predict)])


if __name__ == '__main__':
  train()
  predict()


