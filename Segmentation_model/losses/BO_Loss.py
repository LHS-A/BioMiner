import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryOptimizationLoss(nn.Module):
    """
    Boundary Optimization Loss (BO-Loss)
    Section 2.3 in the paper
    """
    
    def __init__(self, lambda_param: float = 0.2, gamma: float = 1.0):
        super().__init__()
        self.lambda_param = lambda_param
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Total loss function from Equation (5):
        L_total = L_Seg + λ(L_pixel_level + L_image_level)
        """
        # Segmentation loss (CE + Dice with equal weighting)
        seg_loss = self.segmentation_loss(pred, target)
        
        # Boundary optimization losses
        pixel_level_loss = self.pixel_level_loss(pred, image)
        image_level_loss = self.image_level_loss(pred, target)
        
        # Total loss
        total_loss = seg_loss + self.lambda_param * (pixel_level_loss + image_level_loss)
        
        return total_loss
    
    def segmentation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        L_Seg: Linear combination of CE loss and Dice loss with equal weighting
        """
        # Cross-entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        
        # Equal weighting as mentioned in paper
        return (ce_loss + dice_loss) / 2
    
    def pixel_level_loss(self, pred: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Pixel-level loss from Equation (3):
        L_pixel_level = Σ ω_ij * |y_i - y_j|
        where ω_ij = exp(-2γ²/|I_i - I_j|²)
        """
        pred_sigmoid = torch.sigmoid(pred)
        batch_size, _, height, width = pred.shape
        
        # Vectorized implementation for 4-neighborhood
        pred_pad = F.pad(pred_sigmoid, (1, 1, 1, 1), mode='replicate')
        image_pad = F.pad(image, (1, 1, 1, 1), mode='replicate')
        
        # Calculate horizontal and vertical differences
        pred_diff_h = torch.abs(pred_pad[:, :, 1:-1, 2:] - pred_pad[:, :, 1:-1, :-2])  # horizontal
        pred_diff_v = torch.abs(pred_pad[:, :, 2:, 1:-1] - pred_pad[:, :, :-2, 1:-1])  # vertical
        
        image_diff_h = torch.abs(image_pad[:, :, 1:-1, 2:] - image_pad[:, :, 1:-1, :-2])
        image_diff_v = torch.abs(image_pad[:, :, 2:, 1:-1] - image_pad[:, :, :-2, 1:-1])
        
        # Calculate weights
        weight_h = torch.exp(-2 * self.gamma ** 2 / (image_diff_h ** 2 + 1e-6))
        weight_v = torch.exp(-2 * self.gamma ** 2 / (image_diff_v ** 2 + 1e-6))
        
        # Weighted loss
        loss_h = (weight_h * pred_diff_h).mean()
        loss_v = (weight_v * pred_diff_v).mean()
        # print("pixel_loss:{}".format((loss_h + loss_v) / 2))
        
        return (loss_h + loss_v) / 2
    
    def image_level_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Image-level loss from Equation (4):
        L_image_level = Σ (E_pred,i/ΣE_pred,j - E_true,i/ΣE_true,j)^2
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Generate boundary maps using Sobel filter
        pred_boundary = self._sobel_filter(pred_sigmoid)
        target_boundary = self._sobel_filter(target)
        
        # Normalize boundary maps (convert to probability distributions)
        pred_sum = pred_boundary.sum(dim=(2, 3), keepdim=True) + 1e-6
        target_sum = target_boundary.sum(dim=(2, 3), keepdim=True) + 1e-6
        
        pred_norm = pred_boundary / pred_sum
        target_norm = target_boundary / target_sum
        
        # Calculate squared differences
        loss = torch.mean((pred_norm - target_norm) ** 2)
        # print("image_loss:{}".format(loss))  

        return loss
    
    def _sobel_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sobel filter to extract boundaries"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(x.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
        
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return boundary

def calculate_bo_loss(pred, target, image, lambda_param=0.2, gamma=1.0):
    """Convenience function to calculate boundary optimization loss"""
    loss_fn = BoundaryOptimizationLoss(lambda_param=lambda_param, gamma=gamma)
    return loss_fn(pred, target, image) 

class DiceLoss(nn.Module):
    def __init__(self, num_classes, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def diceCoeff(self, pred, gt, eps=1e-5):
        r""" computational formula：
            dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
        """
        N = gt.size(0)
        pred = torch.sigmoid(pred).float()
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)

        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        loss =  (2 * intersection + eps) / (unionset + eps)
        loss = loss.clone().detach().requires_grad_(True)
        return loss.sum() / N

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(self.num_classes):
            class_dice.append(self.diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
    
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice
