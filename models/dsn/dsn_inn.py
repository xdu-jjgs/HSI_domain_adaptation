import torch
import torch.nn as nn
from typing import List

from models.modules import Gate
from models.backbone import ImageClassifier, SingleLayerClassifier
from models.utils.init import initialize_weights
from tllib.modules.grl import GradientReverseLayer, WarmStartGradientReverseLayer


class DSN_INN(nn.Module):
    # TODO: denoise autoencoder
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int):
        super(DSN_INN, self).__init__()
        assert patch_size in [11]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 512
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, 3, 1, 1),
            nn.Upsample(size=(3, 3)),  # 1*1 ==> 3*3
            nn.BatchNorm2d(256),
            self.relu,

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=(2, 2)),  # 3*3 ==> 6*6
            nn.BatchNorm2d(128),
            self.relu,

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Upsample(size=(11, 11)),  # 6*6 ==> 12*12
            nn.BatchNorm2d(64),
            self.relu,

            nn.Conv2d(64, self.num_channels, 3, 1, 1)
            # nn.Upsample(size=(23, 23)) # 12*12 ==> 23*23
        )
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        decoder_output = self.shared_decoder(torch.add(shared_features, private_features))
        class_output = self.classifier(shared_features)[-1]
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        return shared_features, private_features, class_output, domain_output, decoder_output


class DSN_INN_ChannelFilter(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int, filter_ratio: float):
        assert 0. <= filter_ratio <= 1.
        super(DSN_INN_ChannelFilter, self).__init__()
        assert patch_size in [11]
        # backbone输入通道数
        self.filter_ratio = filter_ratio
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 512
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, 3, 1, 1),
            nn.Upsample(size=(3, 3)),  # 1*1 ==> 3*3
            nn.BatchNorm2d(256),
            self.relu,

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=(2, 2)),  # 3*3 ==> 6*6
            nn.BatchNorm2d(128),
            self.relu,

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Upsample(size=(11, 11)),  # 6*6 ==> 12*12
            nn.BatchNorm2d(64),
            self.relu,

            nn.Conv2d(64, self.num_channels, 3, 1, 1)
            # nn.Upsample(size=(23, 23)) # 12*12 ==> 23*23
        )
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = SingleLayerClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind, channel_filter: bool = False):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        class_output = self.classifier(shared_features)[-1]
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]

        mask = torch.ones_like(shared_features).to(shared_features.device)
        if channel_filter:
            scores = self.domain_discriminator.cal_scores(reverse_shared_features, task_ind)
            filter_num = int(self.filter_ratio * scores.size()[1])
            threshold, index = scores.topk(filter_num)
            # print("filter num:{}/{} {} {} {}".format(filter_num, scores.size()[1], scores.size(), index.size(),
            # mask.size()))
            row = torch.arange(mask.size()[0]).unsqueeze(1).to(mask.device)
            mask[row, index] = 0.
        masked_shared_features = shared_features * mask
        decoder_output = self.shared_decoder(torch.add(masked_shared_features, private_features))
        # print(shared_features.size(), mask.size(), masked_shared_features.size(), private_features.size(),
        # decoder_output.size())
        return shared_features, private_features, class_output, domain_output, decoder_output


class DSN_INN_Grad_ChannelFilter(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int, filter_ratio: float):
        assert 0. <= filter_ratio <= 1.
        super(DSN_INN_Grad_ChannelFilter, self).__init__()
        assert patch_size in [11]
        # backbone输入通道数
        self.filter_ratio = filter_ratio
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 512
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, 3, 1, 1),
            nn.Upsample(size=(3, 3)),  # 1*1 ==> 3*3
            nn.BatchNorm2d(256),
            self.relu,

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=(2, 2)),  # 3*3 ==> 6*6
            nn.BatchNorm2d(128),
            self.relu,

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Upsample(size=(11, 11)),  # 6*6 ==> 12*12
            nn.BatchNorm2d(64),
            self.relu,

            nn.Conv2d(64, self.num_channels, 3, 1, 1)
            # nn.Upsample(size=(23, 23)) # 12*12 ==> 23*23
        )
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def cal_scores(self, features, out, task_ind: int):
        assert task_ind in [1, 2]  # 1 for source domain, 2 for target domain
        out_class = out[:, task_ind - 1].sum()
        grads = torch.autograd.grad(out_class, features, retain_graph=True)[0]
        features.squeeze()
        scores = torch.mul(grads, features)
        scores = scores.squeeze()
        # print(grads)
        return scores

    def forward(self, x, task_ind, channel_filter: bool = False):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        class_output = self.classifier(shared_features)[-1]
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        mask = torch.ones_like(shared_features).to(shared_features.device)
        if channel_filter:
            scores = self.cal_scores(reverse_shared_features, domain_output, task_ind)
            filter_num = int(self.filter_ratio * scores.size()[1])
            threshold, index = scores.topk(filter_num)
            # print("filter num:{}/{} {} {}".format(filter_num, scores.size()[1], index.size(), mask.size()))
            # print(scores)
            row = torch.arange(mask.size()[0]).unsqueeze(1).to(mask.device)
            # print(mask[row, index].size())
            mask[row, index] = 0.
        masked_shared_features = shared_features * mask

        decoder_output = self.shared_decoder(torch.add(masked_shared_features, private_features))
        # print(shared_features.size(), mask.size(), masked_shared_features.size(), private_features.size(),
        # decoder_output.size())
        return shared_features, private_features, class_output, domain_output, decoder_output


class DSN_INN_Gate(DSN_INN):
    def __init__(self, num_classes: int, experts: List[nn.Module], patch_size: int):
        super(DSN_INN_Gate, self).__init__(num_classes, experts, patch_size)
        self.num_task = 2
        self.gates = nn.ModuleList([Gate(self.num_channels, 2) for _ in range(self.num_task)])

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            # TODO:check the input of gate
            task_weight = self.gates[0](x)[-1].softmax(dim=1).unsqueeze(1)
            private_features = self.private_source_encoder(features)
        else:
            task_weight = self.gates[1](x)[-1].softmax(dim=1).unsqueeze(1)
            private_features = self.private_target_encoder(features)
        experts_features = torch.stack([shared_features, private_features], 1)
        experts_features = torch.squeeze(experts_features)
        features_fusion = torch.matmul(task_weight, experts_features)
        features_fusion = features_fusion.view(features_fusion.size()[0], self.out_channels, 1, 1)
        decoder_output = self.shared_decoder(features_fusion)
        class_output = self.classifier(shared_features)[-1]
        reverse_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_features)[-1]
        return shared_features, private_features, class_output, domain_output, decoder_output, task_weight


class DSN_INN_NoDecoder(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int):
        super(DSN_INN_NoDecoder, self).__init__()
        assert patch_size in [11]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 512
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        class_output = self.classifier(shared_features)[-1]
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        return shared_features, private_features, class_output, domain_output


class DSN_INN_NoDecoder_Nospec(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int):
        super(DSN_INN_NoDecoder_Nospec, self).__init__()
        assert patch_size in [11]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 512
        self.private_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        private_features = self.private_encoder(features)
        class_output = self.classifier(shared_features)[-1]
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        return shared_features, private_features, class_output, domain_output


class DSN_INN_NoDecoder_NoDis(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int):
        super(DSN_INN_NoDecoder_NoDis, self).__init__()
        assert patch_size in [11]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 512
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        initialize_weights(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output


class DSN_INN_NoDecoder_DST(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int):
        super(DSN_INN_NoDecoder_DST, self).__init__()
        assert patch_size in [11]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 512
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, 512, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.classifier_adv = ImageClassifier(self.out_channels, num_classes)
        self.classifier_pse = ImageClassifier(self.out_channels, num_classes)
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=200, auto_step=True)
        initialize_weights(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)

        out = self.classifier(shared_features)[-1]
        out_pse = self.classifier_pse(shared_features)[-1]
        reverse_shared_features = self.grl_layer(shared_features)
        out_adv = self.classifier_adv(reverse_shared_features)[-1]
        return shared_features, private_features, out, out_pse, out_adv
