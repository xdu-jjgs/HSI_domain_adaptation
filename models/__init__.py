from configs import CFG
from models.ddc.ddc import *
from .dann import DANN
from models.backbone.generator import *
from models.backbone.extractor import FeatureExtractor
from models.backbone.classifier import ImageClassifier
from models.backbone.discriminator import Discriminator
from models.backbone.resnet import ResNet
from models.backbone.capsnet import CapsuleNet
from models.backbone.rescaps import ResCaps


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
    elif CFG.MODEL.NAME == 'pixelda-image':
        G = GeneratorImage(num_channels, num_classes)
        domain_discriminator = DDC(num_channels, num_classes=2)
        class_dicriminator = ResNet(num_channels, num_classes, depth=18, pretrained=False)
        return G, domain_discriminator, class_dicriminator
    elif CFG.MODEL.NAME == 'pixelda-baseline':
        G = GeneratorImage(num_channels, num_classes)
        domain_discriminator = DDC(num_channels, num_classes=2)
        class_dicriminator = ResNet(num_channels, num_classes, depth=18, pretrained=False)
        return G, domain_discriminator, class_dicriminator
    elif CFG.MODEL.NAME == 'pixelda-condition':
        G = GeneratorImage(num_channels, num_classes)
        domain_discriminator = DDCondition(num_channels, num_classes=2, condition_l=num_classes)
        class_dicriminator = ResNet(num_channels, num_classes, depth=18, pretrained=False)
        return G, domain_discriminator, class_dicriminator
    elif CFG.MODEL.NAME == 'pixelda-caps':
        G = GeneratorImage(num_channels, num_classes)
        domain_discriminator = DDC(num_channels, num_classes=2)
        # class_dicriminator = CapsuleNet(num_channels, num_classes)
        class_dicriminator = ResCaps(num_channels, num_classes, depth=18, pretrained=False)
        return G, domain_discriminator, class_dicriminator
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
