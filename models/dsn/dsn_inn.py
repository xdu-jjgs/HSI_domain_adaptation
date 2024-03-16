import torch
import torch.nn as nn
from typing import List

from models.modules import Gate, Decoder
from models.backbone import ImageClassifier, SingleLayerClassifier
from models.utils.init import initialize_weights
from models.utils.grad_filter import cal_grad_scores, mask_channels
from tllib.modules.grl import GradientReverseLayer, WarmStartGradientReverseLayer


class DSN_INN(nn.Module):
    # TODO: denoise autoencoder
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int, filter_ratio: float = None):
        super(DSN_INN, self).__init__()
        if filter_ratio:
            assert 0. <= filter_ratio <= 1.
            self.filter_ratio = filter_ratio
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 128
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_decoder = Decoder(self.out_channels, self.num_channels, patch_size)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind, channel_filter: bool = False, use_abs=False):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        if channel_filter:
            assert self.filter_ratio > 0.
            scores = cal_grad_scores(reverse_shared_features, domain_output, task_ind, use_abs)
            filter_num = int(self.filter_ratio * scores.size()[1])
            masked_ds_shared_features = mask_channels(shared_features, scores, filter_num)
            # masked_ds_private_features = private_features * mask_ds
            # mask_di = torch.ones_like(shared_features).to(shared_features.device)
            # _, index_di_channels = scores.topk(filter_num, largest=False)
            # mask_di[row, index_di_channels] = 0.
            # masked_di_shared_features = shared_features * mask_di
            decoder_output = self.shared_decoder(torch.add(masked_ds_shared_features, private_features))
            class_output = self.classifier(masked_ds_shared_features)[-1]
        else:
            decoder_output = self.shared_decoder(torch.add(shared_features, private_features))
            class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output, domain_output, decoder_output


class DSN_INN_NoDis(nn.Module):
    # TODO: denoise autoencoder
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int):
        super(DSN_INN_NoDis, self).__init__()
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 128
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_decoder = Decoder(self.out_channels, self.num_channels, patch_size)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
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
        return shared_features, private_features, class_output, decoder_output


class DSN_INN_ChannelFilter(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int, hyperparams: List):
        super(DSN_INN_ChannelFilter, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 128
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_decoder = Decoder(self.out_channels, self.num_channels, patch_size)
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
            _, index_ds_channels = scores.topk(filter_num)
            _, index_di_channels = scores.topk(filter_num, largest=False)

            # print("filter num:{}/{} {} {} {}".format(filter_num, scores.size()[1], scores.size(), index_ds_channels.size(),
            # mask.size()))
            row = torch.arange(mask.size()[0]).unsqueeze(1).to(mask.device)
            mask[row, index_ds_channels] = 0.
        masked_shared_features = shared_features * mask
        decoder_output = self.shared_decoder(torch.add(masked_shared_features, private_features))
        # print(shared_features.size(), mask.size(), masked_shared_features.size(), private_features.size(),
        # decoder_output.size())
        return shared_features, private_features, class_output, domain_output, decoder_output


# unused
class DSN_INN_Grad_ChannelFilter(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int, hyperparams: List):
        super(DSN_INN_Grad_ChannelFilter, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 128
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_decoder = Decoder(self.out_channels, self.num_channels, patch_size)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind, channel_filter: bool = False, use_abs=False):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)

        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        if channel_filter:
            scores = cal_grad_scores(reverse_shared_features, domain_output, task_ind, use_abs)
            filter_num = int(self.filter_ratio * scores.size()[1])

            # print("filter num:{}/{} {} {}".format(filter_num, scores.size()[1], index_ds_channels.size(),
            # mask_ds.size())) print(scores)
            mask_ds = torch.ones_like(shared_features).to(shared_features.device)
            row = torch.arange(mask_ds.size()[0]).unsqueeze(1).to(mask_ds.device)
            _, index_ds_channels = scores.topk(filter_num)
            mask_ds[row, index_ds_channels] = 0.
            masked_ds_shared_features = shared_features * mask_ds
            # masked_ds_private_features = private_features * mask_ds

            # mask_di = torch.ones_like(shared_features).to(shared_features.device)
            # _, index_di_channels = scores.topk(filter_num, largest=False)
            # mask_di[row, index_di_channels] = 0.
            # masked_di_shared_features = shared_features * mask_di

            class_output = self.classifier(masked_ds_shared_features)[-1]
            decoder_output = self.shared_decoder(torch.add(masked_ds_shared_features, private_features))
        else:
            class_output = self.classifier(shared_features)[-1]
            decoder_output = self.shared_decoder(torch.add(shared_features, private_features))

        # print(shared_features.size(), mask_ds.size(), masked_ds_shared_features.size(), private_features.size(),
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
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List = None):
        super(DSN_INN_NoDecoder, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 128
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind, channel_filter: bool = False, use_abs=False):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        if channel_filter:
            assert self.filter_ratio > 0.
            scores = cal_grad_scores(reverse_shared_features, domain_output, task_ind, use_abs)
            filter_num = int(self.filter_ratio * scores.size()[1])
            masked_ds_shared_features = mask_channels(shared_features, scores, filter_num)
            class_output = self.classifier(masked_ds_shared_features)[-1]
            # torch.save(masked_ds_shared_features, 'masked_ds_shared_features.pt')
            # raise NotImplementedError
        else:
            class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output, domain_output


class DSN_INN_NoDecoder_NoSpec(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List = None):
        super(DSN_INN_NoDecoder_NoSpec, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = backbone.out_channels
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind, mask_ds: bool = False, use_abs=False, mask_di: bool = False, cal_std: bool = False):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        private_features = features - shared_features
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        if mask_ds:
            assert 0. <= self.filter_ratio <= 1.
            scores = cal_grad_scores(reverse_shared_features, domain_output, task_ind, use_abs)
            filter_num = int(self.filter_ratio * scores.size()[1])
            masked_shared_features = mask_channels(shared_features, scores, filter_num)
            class_output = self.classifier(masked_shared_features)[-1]
            if mask_di:
                scores_abs = scores.abs()
                masked_private_features = mask_channels(private_features, scores_abs, filter_num, largest=False)
                if cal_std:
                    return masked_shared_features, masked_private_features, class_output, domain_output, \
                        shared_features, private_features
                else:
                    return masked_shared_features, masked_private_features, class_output, domain_output
        else:
            class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output, domain_output


class DSN_INN_NoDecoder_NoDis(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List):
        super(DSN_INN_NoDecoder_NoDis, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 128
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        initialize_weights(self)

    def forward(self, x, task_ind, channel_filter: bool = False, scores=None):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        if channel_filter:
            assert scores is not None
            filter_num = int(self.filter_ratio * scores.size()[1])
            mask_ds = torch.ones_like(shared_features).to(shared_features.device)
            row = torch.arange(mask_ds.size()[0]).unsqueeze(1).to(mask_ds.device)
            _, index_ds_channels = scores.topk(filter_num)
            mask_ds[row, index_ds_channels] = 0.
            masked_ds_shared_features = shared_features * mask_ds
            class_output = self.classifier(masked_ds_shared_features)[-1]
        else:
            class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output


class DSN_INN_NoDecoder_NoSpec_NoDis(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(DSN_INN_NoDecoder_NoSpec_NoDis, self).__init__()
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = backbone.out_channels
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        private_features = features - shared_features
        class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output


class DSN_INN_NoDecoder_NoCls(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List):
        super(DSN_INN_NoDecoder_NoCls, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 128
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        initialize_weights(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        if task_ind == 1:
            private_features = self.private_source_encoder(features)
        else:
            private_features = self.private_target_encoder(features)
        return shared_features, private_features


class DSN_INN_NoDecoder_NoSpec_NoCis(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List):
        super(DSN_INN_NoDecoder_NoSpec_NoCis, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = backbone.out_channels
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        private_features = features - shared_features
        return shared_features, private_features


class DSN_INN_NoDecoder_DST(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(DSN_INN_NoDecoder_DST, self).__init__()
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = 128
        self.private_source_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.private_target_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
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
