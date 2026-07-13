import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from ChallengeDL.dataset import DepthDataset
from ChallengeDL.utils import visualize_img, ssim
from ChallengeDL.model import Model
import random

class Solver():

    def __init__(self, args):
        # prepare a dataset
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform_augmentation = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.Resize(232),
            T.CenterCrop(224)
        ])
        self.transform_only = T.Compose([
            T.Resize(232),
            T.CenterCrop(224)
        ])
        self.net = Model().to(self.device)
        # normalization for rgb image only based on resnet pre-training
        self.normalize_rgb = T.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])

        if self.args.is_train:
            self.train_data = DepthDataset(train=DepthDataset.TRAIN,
                                           data_dir=args.data_dir,
                                           transform=self.transform_augmentation)
            self.val_data = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=args.data_dir,
                                         transform=self.transform_only)

            self.train_loader = DataLoader(dataset=self.train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True)

            decoder_params = list(self.net.decoder.parameters())

            self.optimizer = torch.optim.Adam([
                {"params": decoder_params, "lr": args.lr, "weight_decay": 1e-3}  # decoder
            ])

            self.l1_loss = L1Loss()
            #print(self.optimizer)

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                eta_min=1e-7,
                T_max=args.max_epochs
            )

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
        else:
            self.test_set = DepthDataset(train=DepthDataset.TEST,
                                    data_dir=self.args.data_dir,
                                         transform=self.transform_only)
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))
            self.net.eval()

    def fit(self):
        args = self.args
        #print(self.net)
        for epoch in range(args.max_epochs):
            self.net.train()

            if epoch == 5:
                print(" Unlocking last encoder layer at epoch", epoch)
                for layer in [self.net.encoder_layer3, self.net.encoder_layer4]:
                    for p in layer.parameters():
                        p.requires_grad = True
                self.optimizer.add_param_group(
                    {"params": self.net.encoder_layer3.parameters(), "lr": 1e-6, "weight_decay": 1e-5})
                self.optimizer.add_param_group(
                    {"params": self.net.encoder_layer4.parameters(), "lr": 1e-6, "weight_decay": 1e-5})
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    eta_min=1e-7,
                    T_max=args.max_epochs
                )
            for step, inputs in enumerate(self.train_loader):
                rgb_image = inputs[0].to(self.device)
                # normalize rgb image only
                rgb_image = self.normalize_rgb(rgb_image)
                depth_image = inputs[1].to(self.device)
                pred = self.net(rgb_image)
                loss, berhu, l_ssim, l_grad = self.combined_loss(pred,depth_image)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("Epoch [{}/{}] Loss: {:.3f} ".
                      format(epoch + 1, args.max_epochs, loss.item()))
                 #print("berhu: {} ssim: {} grad: {}".format(berhu,l_ssim,l_grad))
            # Valutazione ad ogni epoca
            self.evaluate(DepthDataset.VAL)
            self.scheduler.step()

            print(self.scheduler.get_last_lr())
            self.save(args.ckpt_dir, args.ckpt_name, epoch + 1)
        return

    def berhu_loss(self, pred, label, threshold= 0.2):
        error = pred - label
        abs_error = torch.abs(error)

        c = threshold * torch.max(abs_error).item()

        l1_loss = abs_error
        l2_loss = (abs_error**2 + c**2) / (2 * c)

        mask = (abs_error <= c)
        loss = torch.where(mask, l1_loss, l2_loss)
        return loss.mean()

    def gradient_loss(self, pred, target):
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        loss = torch.mean(torch.abs(dy_pred - dy_target)) + torch.mean(torch.abs(dx_pred - dx_target))
        return loss

    def combined_loss(self, pred, depth_image):
        l_berhu = 0.2 * self.berhu_loss(pred, depth_image)
        #l1 = 0.1 * self.l1_loss(pred, depth_image)
        l_ssim = 0.5 * ((1 - ssim(pred,depth_image)) / 2 )
        l_grad = 0.5 * (self.gradient_loss(pred,depth_image))
        loss = l_berhu +  l_ssim + l_grad
        return loss, l_berhu, l_ssim, l_grad

    def evaluate(self, set):

        args = self.args
        if set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
        elif set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        loss_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                # apply normalization only to rgb image
                images_norm = self.normalize_rgb(images)
                output= self.net(images_norm.to(self.device))
                loss_comb, l_berhu, l_ssim, l_grad = self.combined_loss(output, depth.to(self.device))
                loss_acc += loss_comb.item()
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                # visualize original image without normalization
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix=suffix)
        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader))
        print("loss on", suffix, ":", loss_acc / len(loader))

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

    def test(self):

        loader = DataLoader(self.test_set,
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                # apply normalization only to rgb image
                images_norm = self.normalize_rgb(images)
                output = self.net(images_norm.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))

