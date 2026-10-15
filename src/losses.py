
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiImplicitCrossEntropy(nn.Module):


    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        ce_loss = self.ce(outputs, labels)
        softmax_out = torch.softmax(outputs, dim=1)

        intra_loss = -torch.mean(torch.sum(softmax_out * torch.log(softmax_out + 1e-8), dim=1))
        return ce_loss + 0.1 * intra_loss


class FocalLoss(nn.Module):


    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, labels):
        ce_loss = nn.functional.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):


    def __init__(self, smoothing=0.1, num_classes=5):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, outputs, labels):
        confidence = 1.0 - self.smoothing
        smooth_label = self.smoothing / (self.num_classes - 1)
        one_hot = torch.zeros_like(outputs).scatter(1, labels.unsqueeze(1), 1)
        smoothed_labels = one_hot * confidence + smooth_label
        log_probs = torch.log_softmax(outputs, dim=1)
        return -torch.mean(torch.sum(smoothed_labels * log_probs, dim=1))


class DiceLoss(nn.Module):


    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, labels):
        outputs = torch.softmax(outputs, dim=1)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=outputs.size(1)).float()


        intersection = torch.sum(outputs * labels_one_hot, dim=1)
        union = torch.sum(outputs + labels_one_hot, dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - torch.mean(dice)


class FrequencyDomainLoss(nn.Module):


    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, inputs):
        ce_loss = self.ce(outputs, labels)


        with torch.no_grad():
            fft_input = torch.fft.fft(inputs.squeeze(1))
            mag_input = torch.abs(fft_input)

        fft_output = torch.fft.fft(outputs.unsqueeze(1))
        mag_output = torch.abs(fft_output)

        freq_loss = F.mse_loss(mag_output, mag_input)
        return ce_loss + self.alpha * freq_loss


def initialize_losses():

    criterion1 = SemiImplicitCrossEntropy()
    criterion2 = FocalLoss(gamma=2.0)
    criterion3 = LabelSmoothingLoss(smoothing=0.1)
    criterion4 = DiceLoss()
    freq_criterion = FrequencyDomainLoss()
    
    loss_functions = [criterion1, criterion2, criterion3, criterion4, freq_criterion]
    loss_weights = [0.25, 0.25, 0.2, 0.15, 0.15]
    
    return loss_functions, loss_weights
