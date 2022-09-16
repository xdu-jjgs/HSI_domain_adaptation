from configs import CFG
from .ddc import DDC
from .dann import DANN
from models.backbone.generator import Generator
from models.backbone.extractor import FeatureExtractor
from models.backbone.classifier import ImageClassifier
from models.backbone.discriminator import Discriminator
from models.backbone.resnet import ResNet


def build_model(num_channels, num_classes):
    if CFG.MODEL.NAME == 'ddc':
        return DDC(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'mcd':
        out_channels = 512
        FE = FeatureExtractor(num_channels, out_channels)
        C1 = Discriminator(out_channels, num_classes)
        C2 = Discriminator(out_channels, num_classes)
        return FE, C1, C2
    elif CFG.MODEL.NAME == 'dann':
        return DANN(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'pixelda':
        G = Generator(num_channels, num_classes)
        domain_discriminator = DDC(num_channels, num_classes=2)
        class_dicriminator = ResNet(num_channels, num_classes+1, depth=18, pretrained=False)
        return G, domain_discriminator, class_dicriminator
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
