import torch.nn.functional as F

class Generator(nn.Module):
  def __init__(self, out_dim=512):
    super(Model,self).__init__()
    self.fc1 = nn.Linear(self.out_dim * 3, self.out_dim)
    self.fc2 = nn.Linear(self.out_dim, self.out_dim)
    self.fc3 = nn.Linear(self.out_dim, self.out_dim)

  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class Embedder(nn.Module):
  def __init__(self, in_dim=512, out_dim=512):
    super(Model,self).__init__()
    self.fc1 = nn.Linear(self.in_dim, self.out_dim)
    self.fc2 = nn.Linear(self.out_dim, self.out_dim)

  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
