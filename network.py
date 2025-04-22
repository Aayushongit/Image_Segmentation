import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        
        gt_boundaries = self.compute_boundaries(gt)
        
        weight_map = torch.exp(-torch.pow(gt_boundaries, 2) / (2 * self.theta * self.theta))
        
        bce = F.binary_cross_entropy(pred, gt, reduction='none')
        weighted_bce = weight_map * bce
        
        return weighted_bce.mean()
    
    def compute_boundaries(self, mask):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(mask.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(mask.device)
        
        mask_x = F.conv2d(mask, sobel_x, padding=1)
        mask_y = F.conv2d(mask, sobel_y, padding=1)
        
        edge = torch.sqrt(torch.pow(mask_x, 2) + torch.pow(mask_y, 2))
        return edge

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, dropout_prob=0.0, use_residual=False):
        super(conv_block, self).__init__()
        self.use_residual = use_residual
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        )
        self.residual_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0) if use_residual else None

    def forward(self, x):
        if self.use_residual:
            residual = self.residual_conv(x)
            return self.conv(x) + residual
        else:
            return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class DeepSupervisionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepSupervisionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        x = x.transpose(1, 2).reshape(b, c, h, w)  # [B, C, H, W]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, dropout_prob=0.0, use_residual=False, use_deep_supervision=False):
        super(U_Net, self).__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64, dropout_prob=dropout_prob, use_residual=use_residual)
        self.Conv2 = conv_block(ch_in=64, ch_out=128, dropout_prob=dropout_prob, use_residual=use_residual)
        self.Conv3 = conv_block(ch_in=128, ch_out=256, dropout_prob=dropout_prob, use_residual=use_residual)
        self.Conv4 = conv_block(ch_in=256, ch_out=512, dropout_prob=dropout_prob, use_residual=use_residual)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024, dropout_prob=dropout_prob, use_residual=use_residual)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, dropout_prob=dropout_prob, use_residual=use_residual)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, dropout_prob=dropout_prob, use_residual=use_residual)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, dropout_prob=dropout_prob, use_residual=use_residual)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, dropout_prob=dropout_prob, use_residual=use_residual)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
        if use_deep_supervision:
            self.dsv4 = DeepSupervisionBlock(512, output_ch)
            self.dsv3 = DeepSupervisionBlock(256, output_ch)
            self.dsv2 = DeepSupervisionBlock(128, output_ch)
            self.dsv1 = DeepSupervisionBlock(64, output_ch)
            
            self.final_dsv = nn.Conv2d(4 * output_ch, output_ch, kernel_size=1)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        if self.use_deep_supervision:
            dsv4 = self.dsv4(d5)
            dsv4 = F.interpolate(dsv4, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv3 = self.dsv3(d4)
            dsv3 = F.interpolate(dsv3, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv2 = self.dsv2(d3)
            dsv2 = F.interpolate(dsv2, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv1 = self.dsv1(d2)
            
            final = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
            d1 = self.final_dsv(final)

        return d1

class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2, dropout_prob=0.0, use_deep_supervision=False):
        super(R2U_Net, self).__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
        if use_deep_supervision:
            self.dsv4 = DeepSupervisionBlock(512, output_ch)
            self.dsv3 = DeepSupervisionBlock(256, output_ch)
            self.dsv2 = DeepSupervisionBlock(128, output_ch)
            self.dsv1 = DeepSupervisionBlock(64, output_ch)
            
            self.final_dsv = nn.Conv2d(4 * output_ch, output_ch, kernel_size=1)


    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        
        if self.use_deep_supervision:
            dsv4 = self.dsv4(d5)
            dsv4 = F.interpolate(dsv4, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv3 = self.dsv3(d4)
            dsv3 = F.interpolate(dsv3, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv2 = self.dsv2(d3)
            dsv2 = F.interpolate(dsv2, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv1 = self.dsv1(d2)
            
            final = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
            d1 = self.final_dsv(final)

        return d1

class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, dropout_prob=0.0, use_residual=False, use_deep_supervision=False):
        super(AttU_Net, self).__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64, dropout_prob=dropout_prob, use_residual=use_residual)
        self.Conv2 = conv_block(ch_in=64, ch_out=128, dropout_prob=dropout_prob, use_residual=use_residual)
        self.Conv3 = conv_block(ch_in=128, ch_out=256, dropout_prob=dropout_prob, use_residual=use_residual)
        self.Conv4 = conv_block(ch_in=256, ch_out=512, dropout_prob=dropout_prob, use_residual=use_residual)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024, dropout_prob=dropout_prob, use_residual=use_residual)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, dropout_prob=dropout_prob, use_residual=use_residual)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, dropout_prob=dropout_prob, use_residual=use_residual)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, dropout_prob=dropout_prob, use_residual=use_residual)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, dropout_prob=dropout_prob, use_residual=use_residual)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
        if use_deep_supervision:
            self.dsv4 = DeepSupervisionBlock(512, output_ch)
            self.dsv3 = DeepSupervisionBlock(256, output_ch)
            self.dsv2 = DeepSupervisionBlock(128, output_ch)
            self.dsv1 = DeepSupervisionBlock(64, output_ch)
            
            self.final_dsv = nn.Conv2d(4 * output_ch, output_ch, kernel_size=1)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        if self.use_deep_supervision:
            dsv4 = self.dsv4(d5)
            dsv4 = F.interpolate(dsv4, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv3 = self.dsv3(d4)
            dsv3 = F.interpolate(dsv3, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv2 = self.dsv2(d3)
            dsv2 = F.interpolate(dsv2, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
            dsv1 = self.dsv1(d2)
            
            final = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
            d1 = self.final_dsv(final)

        return d1

class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2, dropout_prob=0.0, use_deep_supervision=False):
        super(R2AttU_Net, self).__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
