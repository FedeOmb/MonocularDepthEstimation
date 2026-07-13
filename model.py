import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # loading resnet50 with weights
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]

        #extract single resnet layers up to layer 4
        self.encoder_layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.encoder_layer1 = resnet.layer1  # 64 -> 256 channels
        self.encoder_layer2 = resnet.layer2  # 256 -> 512 channels
        self.encoder_layer3 = resnet.layer3  # 512 -> 1024 channels
        self.encoder_layer4 = resnet.layer4  # 1024 -> 2048 channels

        # upscaling output height to input / 16
        self.decoder_up0_1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        # concatenate with layer3 encoder output reduced to 512 channels
        self.reduce_enc3 = nn.Conv2d(1024, 512, 1)
        self.decoder_layer1 = nn.Sequential(
            nn.Conv2d(1024+512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder_up1_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # upscaling output height to input / 8
        # concatenate with layer2 encoder output reduced to 256 channels
        self.reduce_enc2 = nn.Conv2d(512, 256, 1)

        self.decoder_layer2 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # upscaling output height to input / 4
        self.decoder_up2_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # concatenate with layer2 encoder output reduced to 128 channels
        self.reduce_enc1 = nn.Conv2d(256, 128, 1)

        self.decoder_layer3 = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # upscaling output height to input / 2
        self.decoder_up3_4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.decoder_layer4 = nn.Sequential( # 128+64 -> 64
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # upscaling output height to input
        self.decoder_up4_5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.decoder_layer5 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.ModuleList([
            self.decoder_up0_1, self.decoder_layer1, self.decoder_up1_2, self.decoder_layer2,
            self.decoder_up2_3, self.decoder_layer3, self.decoder_up3_4, self.decoder_layer4,
            self.decoder_up4_5, self.decoder_layer5, self.out_conv, self.reduce_enc1,
            self.reduce_enc2, self.reduce_enc3
        ])

        # lock training of resnet encoder
        for layer in [self.encoder_layer0, self.encoder_layer1, self.encoder_layer2, self.encoder_layer3, self.encoder_layer4]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, x):

        # encoder forward
        enc0_out = self.encoder_layer0(x)
        enc1_out = self.encoder_layer1(enc0_out)
        enc2_out = self.encoder_layer2(enc1_out)
        enc3_out = self.encoder_layer3(enc2_out)
        enc4_out = self.encoder_layer4(enc3_out)

        # decoder forward

        d0_out_up = self.decoder_up0_1(enc4_out)
        # concatenate with layer3 encoder output
        enc3_reduced = self.reduce_enc3(enc3_out)
        d0_out_concat = torch.cat([d0_out_up, enc3_reduced], dim=1)
        d1_out = self.decoder_layer1(d0_out_concat)
        #upscaling
        d1_out_up = self.decoder_up1_2(d1_out)
        # concatenate with layer2 encoder output
        enc2_reduced = self.reduce_enc2(enc2_out)
        d1_out_concat = torch.cat([d1_out_up, enc2_reduced], dim=1)
        d2_out = self.decoder_layer2(d1_out_concat)
        #upscaling
        d2_out_up = self.decoder_up2_3(d2_out)
        # concatenate with layer1 encoder output
        enc1_reduced = self.reduce_enc1(enc1_out)
        d2_out_concat = torch.cat([d2_out_up, enc1_reduced], dim=1)
        d3_out = self.decoder_layer3(d2_out_concat)
        #upscaling
        d3_out_up = self.decoder_up3_4(d3_out)
        d4_out = self.decoder_layer4(d3_out_up)
        d4_out_up = self.decoder_up4_5(d4_out)
        d5_out = self.decoder_layer5(d4_out_up)
        out = self.out_conv(d5_out)

        return out