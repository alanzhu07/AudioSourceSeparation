import torch
import torch.nn as nn
import transforms

class L1FreqLoss(nn.Module):
    def __init__(self):
        super(L1FreqLoss, self).__init__()
    
    def forward(self, outputs, target):
        outputs = transforms.stft(outputs)
        target = transforms.stft(target)

        outputs = outputs.view(-1)
        target = target.view(-1)

        loss = torch.mean(torch.abs(torch.abs(outputs) - torch.abs(target)))
        return loss
