import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, alexnet, resnet18, googlenet


class BinaryClassifier(nn.Module):
    """Binary Classification Model"""
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 3256),
            nn.ReLU(),
            nn.Linear(3256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def get_binaryclassmodel():
    return BinaryClassifier()


class MulticlassClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MulticlassClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def get_multiclassmodel(num_classes):
    return MulticlassClassifier(num_classes)


def get_optimizer(optimizer_name, model, lr=0.001):
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name. Choose 'SGD', 'Adam', or 'RMSprop'.")


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = None
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 512).to(x.device)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_CNN():
    return CNNModel()


class VGG16Model(nn.Module):
    """VGG16 Model for Classification"""
    def __init__(self, num_classes):
        super(VGG16Model, self).__init__()
        self.model = vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class AlexNetModel(nn.Module):
    """AlexNet Model for Classification"""
    def __init__(self, num_classes):
        super(AlexNetModel, self).__init__()
        self.model = alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNetModel(nn.Module):
    """ResNet Model for Classification"""
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class GoogLeNetModel(nn.Module):
    """GoogLeNet Model for Classification"""
    def __init__(self, num_classes):
        super(GoogLeNetModel, self).__init__()
        self.model = googlenet(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    """Autoencoder Model for Feature Extraction"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_autoencodermodel():
    return Autoencoder()


class EncoderWithResNetClassifier(nn.Module):
    """Combines Autoencoder Encoder with ResNet Classifier"""
    def __init__(self, num_classes):
        super(EncoderWithResNetClassifier, self).__init__()

        self.encoder = Autoencoder().encoder
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Identity()
        self.resnet.fc = nn.Linear(256 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.resnet.fc(x)
        return x

def get_encoder_with_resnet(num_classes):
    return EncoderWithResNetClassifier(num_classes)
