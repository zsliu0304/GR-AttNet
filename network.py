import torch
import torch.nn as nn
import torch.nn.functional as F

class GRAttNet(nn.Module):
    def __init__(self):
        super(GRAttNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(4, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Attention
        self.attention = GRAttention(channel=128, kernels=[3, 7], reduction=16, group=1, L=32)
        
        # Residual blocks
        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        
        # Decoder
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4)

        # Heads
        self.position = nn.Conv2d(32, 1, kernel_size=2)
        self.cosine = nn.Conv2d(32, 1, kernel_size=2)
        self.sine = nn.Conv2d(32, 1, kernel_size=2)
        self.width = nn.Conv2d(32, 1, kernel_size=2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.attention(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        return self.position(x), self.cosine(x), self.sine(x), self.width(x)

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        losses = {k: F.smooth_l1_loss(pred, gt) for k, pred, gt in
                  zip(('p_loss', 'cos_loss', 'sin_loss', 'width_loss'),
                      (pos_pred, cos_pred, sin_pred, width_pred),
                      (y_pos, y_cos, y_sin, y_width))}

        return {
            'loss': sum(losses.values()),
            'losses': losses,
            'pred': {'pos': pos_pred, 'cos': cos_pred, 'sin': sin_pred, 'width': width_pred}
        }

    def predict(self, xc):
        pos, cos, sin, width = self(xc)
        return {'pos': pos, 'cos': cos, 'sin': sin, 'width': width}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + x_in)

# This portion of the code is still being organized; 
# we plan to release it publicly once the associated paper has been accepted.

class GRAttention(nn.Module):
    def __init__(self, channel=128, kernels=[3, 7], reduction=16, group=1, L=32):
        super().__init__()

        self.softmax = nn.Softmax(dim=0)




net = GRAttNet()