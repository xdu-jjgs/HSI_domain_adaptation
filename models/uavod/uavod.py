import torch
import torch.nn as nn

from models.utils.init import initialize_weights
from tllib.modules.grl import GradientReverseLayer
from models.backbone.classifier import ImageClassifier


class UAVOD(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(UAVOD, self).__init__()
        self.backbone = backbone
        self.relu = nn.LeakyReLU()
        self.out_channels = backbone.out_channels
        self.fft_conv_di = nn.Conv2d(backbone.in_channels, backbone.in_channels, 1, 1, 0, bias=False)
        self.fft_conv_ds = nn.Conv2d(backbone.in_channels, backbone.in_channels, 1, 1, 0, bias=False)
        self.classifier = ImageClassifier(backbone.out_channels, num_classes)

        initialize_weights(self.fft_conv_di)
        initialize_weights(self.fft_conv_ds)
        initialize_weights(self.classifier)

    def forward(self, x):
        freq_tensor = torch.fft.fft2(x)
        amplitude = torch.abs(freq_tensor)
        phase = torch.angle(freq_tensor)
        amplitude_extract_di = self.fft_conv_di(amplitude)
        complex_tensor_di = amplitude_extract_di * (torch.cos(phase) + 1j * torch.sin(phase))
        recon_di = torch.fft.ifft2(complex_tensor_di).real

        amplitude_extract_ds = self.fft_conv_ds(amplitude)
        complex_tensor_ds = amplitude_extract_ds * (torch.cos(phase) + 1j * torch.sin(phase))
        recon_ds = torch.fft.ifft2(complex_tensor_ds).real

        di_features = self.backbone(recon_di)
        ds_features = self.backbone(recon_ds)
        class_output = self.classifier(di_features)[-1]
        return di_features, ds_features, class_output
