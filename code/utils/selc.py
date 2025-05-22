import torch
import torch.nn.functional as F
import numpy as np

class InverseCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(InverseCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels, weights):
        probs = F.softmax(logits, dim=1)

        loss_correct = self.CEloss(logits, labels)
        loss_incorrect = -torch.log(1 - probs[torch.arange(len(labels)), labels] + 1e-12)
        loss = weights * loss_correct + (1 - weights) * loss_incorrect

        return loss


class SELCLoss(torch.nn.Module):
    def __init__(self, labels, num_classes, es=10, momentum=0.9, reduction='none', weights=None):
        super(SELCLoss, self).__init__()
        self.num_classes = num_classes
        self.soft_labels = torch.zeros(len(labels), num_classes, dtype=torch.float).cuda()
        self.soft_labels[torch.arange(len(labels)), labels] = 1
        self.es = es
        self.momentum = momentum
        self.reduction = reduction
        self.CEloss = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.InverseCEloss = InverseCrossEntropyLoss(num_classes)
        self.weights = weights
            
        if weights is not None:
            assert reduction == 'none'

    def forward(self, logits, labels, index, epoch):
        if epoch <= self.es:
            if self.weights is None:
                return self.CEloss(logits, labels)
            batch_weights = self.weights[index]
            # batch_weights[batch_weights < 0.5] = 0
            # batch_weights[batch_weights >= 0.5] = 1
            batch_weights = batch_weights / np.sum(batch_weights) * len(batch_weights)
            
            batch_weights = torch.tensor(batch_weights, requires_grad=False).cuda()
            
            return self.CEloss(logits, labels) * batch_weights
            # return self.InverseCEloss(logits, labels, batch_weights)
        else:
            pred = F.softmax(logits, dim=1)
            pred_detach = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] \
                                      + (1 - self.momentum) * pred_detach

            selc_loss = -torch.sum(torch.log(pred) * self.soft_labels[index], dim=1)
            if self.reduction == 'none':
                return selc_loss
            elif self.reduction == 'mean':
                return torch.mean(selc_loss)
            elif self.reduction == 'sum':
                return torch.sum(selc_loss)
            else:
                raise ValueError(f"Invalid reduction mode: {self.reduction}")
            
            