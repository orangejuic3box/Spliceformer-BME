# +
import torch
from typing import Optional, Sequence

class categorical_crossentropy_2d:
    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        mask: bool = False,
    ):
        self.mask = mask
        self.eps = torch.finfo(torch.float32).eps

        if self.mask:
            # If mask=True but no weights provided, default to equal weights
            if weights is None:
                self.weights = torch.ones(3, dtype=torch.float32)
            else:
                self.weights = torch.as_tensor(weights, dtype=torch.float32)
        else:
            # When mask=False we don't use weights at all
            self.weights = None

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.mask:
            # self.weights is guaranteed to be a tensor here
            w0, w1, w2 = self.weights[0], self.weights[1], self.weights[2]

            loss_sum = torch.sum(
                w0 * y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + self.eps)
                + w1 * y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + self.eps)
                + w2 * y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + self.eps)
            )

            weight_sum = torch.sum(
                w0 * y_true[:, 0, :]
                + w1 * y_true[:, 1, :]
                + w2 * y_true[:, 2, :]
            ) + self.eps

            return -loss_sum / weight_sum
        else:
            # Standard unweighted categorical cross-entropy,
            # normalized by total "mass" in y_true
            prob_sum = torch.sum(y_true) + self.eps
            ce = -torch.sum(
                y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + self.eps)
                + y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + self.eps)
                + y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + self.eps)
            ) / prob_sum
            return ce

        
class binary_crossentropy_2d:
    def __init__(self):
        self.eps = torch.finfo(torch.float32).eps
        
    def loss(self,y_pred,y_true):
        loss = torch.mean(y_true*torch.log(y_pred+self.eps) + (1-y_true)*torch.log(1-y_pred+self.eps))
        return -loss
    


# -

class kl_div_2d:
    def __init__(self,temp=1):
        self.eps = torch.finfo(torch.float32).eps
        self.temp = temp
        
    def loss(self,y_pred,y_true):
        if self.temp!=1:
            y_true = torch.nn.Softmax(dim=1)(torch.log(y_true+self.eps)/self.temp)
        return -torch.mean((y_true[:, 0, :]*torch.log(y_pred[:, 0, :]/(y_true[:, 0, :]+self.eps)+self.eps) + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]/(y_true[:, 1, :]+self.eps)+self.eps) + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]/(y_true[:, 2, :]+self.eps)+self.eps))*self.temp**2)
        #x = -torch.sum(y_true[:, 0, :]*torch.log(y_pred[:, 0, :]/(y_true[:, 0, :]+self.eps)+self.eps) + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]/(y_true[:, 1, :]+self.eps)+self.eps) + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]/(y_true[:, 2, :]+self.eps)+self.eps))*self.temp**2
        #return x/(y_pred.shape[0]*y_pred.shape[2])
