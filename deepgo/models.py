import torch

__all__ = ["Dual", "dual39", "dual79"]

class _ConvBlock(torch.nn.Module):
    def __init__(self, filters=256):
        super(_ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(17, filters, kernel_size=3, stride=1, padding=1)
        self.norm = torch.nn.BatchNorm2d(filters)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class _ResidualBlock(torch.nn.Module):
    def __init__(self, filters=256):
        super(_ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(filters)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(filters)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + identity
        x = self.relu2(x)
        return x

class _PolicyHead(torch.nn.Module):
    def __init__(self, filters=256, channels=2):
        super(_PolicyHead, self).__init__()
        self.conv = torch.nn.Conv2d(filters, channels, kernel_size=1, stride=1)
        self.norm = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(19 * 19 * channels, 19 * 19 + 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class _ValueHead(torch.nn.Module):
    def __init__(self, filters=256, hidden_dim=256):
        super(_ValueHead, self).__init__()
        self.conv = torch.nn.Conv2d(filters, 1, kernel_size=1, stride=1)
        self.norm = torch.nn.BatchNorm2d(1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(19 * 19 * 1, hidden_dim)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

class Dual(torch.nn.Module):
    def __init__(self, layers, filters=256, policy_channels=2, value_hidden_dim=256):
        super(Dual, self).__init__()
        self.conv = _ConvBlock(filters)
        self.layer = torch.nn.Sequential(*[_ResidualBlock(filters) for _ in range(layers)])
        self.policy = _PolicyHead(filters, policy_channels)
        self.value = _ValueHead(filters, value_hidden_dim)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        p = self.policy(x)
        v = self.value(x)
        return p, v

def dual39():
    return Dual(19)

def dual79():
    return Dual(39)
