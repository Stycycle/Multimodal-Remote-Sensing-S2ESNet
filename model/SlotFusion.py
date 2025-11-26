import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Spatial_Enhance_Module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, size=None):
        """Implementation of SAEM: Spatial Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block if not specifed reduced to half
        """
        super(Spatial_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # dimension == 2
        conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=size * size,
                out_channels=1,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x1, x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels, -1)
        t1 = t1.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)  # B*HW*TF --> B*TF*HW
        Affinity_M = self.dim_reduce(Affinity_M)  # B*1*HW
        Affinity_M = Affinity_M.view(batch_size, 1, x1.size(2), x1.size(3))   # B*1*H*W

        x1 = x1 * Affinity_M.expand_as(x1)

        return x1


class Spectral_Enhance_Module(nn.Module):
    def __init__(self, in_channels, in_channels2, inter_channels=None, inter_channels2=None):
        """Implementation of SEEM: Spectral Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block
        """
        super(Spectral_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.in_channels2 = in_channels2
        self.inter_channels2 = inter_channels2

        if self.inter_channels is None:
            self.inter_channels = in_channels
            if self.inter_channels == 0:
                self.inter_channels = 1
        if self.inter_channels2 is None:
            self.inter_channels2 = in_channels2
            if self.inter_channels2 == 0:
                self.inter_channels2 = 1

        # dimension == 2
        conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels2, out_channels=self.inter_channels2, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels2),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels2,
                out_channels=1,
                kernel_size=1,
                bias=False,
            )
        )

    def forward(self, x1, x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels2, -1)
        t2 = t2.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)  # B*C1*C2 --> B*C2*C1
        Affinity_M = self.dim_reduce(Affinity_M)  # B*1*C1
        Affinity_M = Affinity_M.view(batch_size, x1.size(1), 1, 1)  # B*C1*1*1

        x1 = x1 * Affinity_M.expand_as(x1)

        return x1

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        return out

class S2ESNet(nn.Module):

    def __init__(self, input_channels, input_channels2, n_classes, patch_size, num_slots=8, num_iterations=3):
        super(S2ESNet, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.planes_a = [128, 64, 32]
        self.planes_b = [8, 16, 32]

        # For image a (7×7×input_channels) --> (7×7×planes_a[0])
        self.conv1_a = conv_bn_relu(input_channels, self.planes_a[0], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×input_channels2) --> (7×7×planes_b[0])
        self.conv1_b = conv_bn_relu(input_channels2, self.planes_b[0], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[0]) --> (7×7×planes_a[1])
        self.conv2_a = conv_bn_relu(self.planes_a[0], self.planes_a[1], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[0]) --> (7×7×planes_b[1])
        self.conv2_b = conv_bn_relu(self.planes_b[0], self.planes_b[1], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[1]) --> (7×7×planes_a[2])
        self.conv3_a = conv_bn_relu(self.planes_a[1], self.planes_a[2], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[1]) --> (7×7×planes_b[2])
        self.conv3_b = conv_bn_relu(self.planes_b[1], self.planes_b[2], kernel_size=3, padding=1, bias=True)

        self.SAEM = Spatial_Enhance_Module(in_channels=self.planes_a[2], inter_channels=self.planes_a[2]//2, size=patch_size)
        self.SEEM = Spectral_Enhance_Module(in_channels=self.planes_b[2], in_channels2=self.planes_a[2])

        self.FusionLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.planes_a[2] * 2,
                out_channels=self.planes_a[2],
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.planes_a[2]),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.planes_a[2], n_classes)

        # Use SlotFusion module to perform slot-based fusion and classification
        self.slot_fusion = SlotFusion(
            input_channels=self.planes_a[2],
            input_channels2=self.planes_b[2],
            n_classes=n_classes,
            patch_size=patch_size,
            num_slots=num_slots,
            num_iterations=num_iterations,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.conv1_a(x1)
        x2 = self.conv1_b(x2)

        x1 = self.conv2_a(x1)
        x2 = self.conv2_b(x2)

        x1 = self.conv3_a(x1)
        x2 = self.conv3_b(x2)

        ss_x1 = self.SAEM(x1, x2)
        ss_x2 = self.SEEM(x2, x1)

        x = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))
        
        # Delegate fusion and classification to SlotFusion (expects two modality tensors)
        x = self.slot_fusion(ss_x1, ss_x2)
        # x = self.avg_pool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


if __name__ == '__main__':
    import torch

    img1 = torch.randn(2, 6, 7, 7)
    img2 = torch.randn(2, 6, 7, 7)

    SAEM = Spatial_Enhance_Module(in_channels=6, inter_channels=6 // 2, size=7)
    out = SAEM(img1, img2)
    print(out)

    SEEM = Spectral_Enhance_Module(in_channels=6, in_channels2=6)
    out = SEEM(img1, img2)
    print(out.shape)

class SlotAttention(nn.Module):
    """Slot Attention module for multimodal remote sensing."""

    def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super(SlotAttention, self).__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # Parameters for Gaussian init (shared by all slots)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_size))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_size))

        # Linear maps for the attention module
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(slot_size, slot_size, bias=False)
        self.project_v = nn.Linear(slot_size, slot_size, bias=False)

        # Slot update functions
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size)
        )

    def forward(self, inputs):
        # inputs has shape [batch_size, num_inputs, inputs_size]
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [batch_size, num_inputs, slot_size]
        v = self.project_v(inputs)  # [batch_size, num_inputs, slot_size]

        # Initialize the slots [batch_size, num_slots, slot_size]
        batch_size = inputs.size(0)
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
            batch_size, self.num_slots, self.slot_size, device=inputs.device)

        # Multiple rounds of attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention
            q = self.project_q(slots)  # [batch_size, num_slots, slot_size]
            q = q * (self.slot_size ** -0.5)  # Normalization
            attn_logits = torch.bmm(k, q.transpose(-2, -1))  # [batch_size, num_inputs, num_slots]
            attn = F.softmax(attn_logits, dim=-1)

            # Weighted mean
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(-2, -1), v)  # [batch_size, num_slots, slot_size]

            # Slot update
            slots = self.gru(updates.view(-1, self.slot_size), slots_prev.view(-1, self.slot_size))
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


def spatial_flatten(x):
    """Flatten spatial dimensions."""
    return x.reshape(x.size(0), x.size(2) * x.size(3), x.size(1))


class SlotFusion(nn.Module):
    """Slot Attention-based fusion for multimodal remote sensing classification."""

    def __init__(self, input_channels, input_channels2, n_classes, patch_size, num_slots=8, num_iterations=3):
        super(SlotFusion, self).__init__()
        self.patch_size = patch_size
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        
        # Feature extraction for both modalities
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(input_channels2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Position embedding
        self.pos_embed = SoftPositionEmbed(64, (patch_size, patch_size))
        
        # Layer norm and MLP for feature processing
        self.layer_norm = nn.LayerNorm(64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Slot attention module
        self.slot_attention = SlotAttention(
            num_iterations=num_iterations,
            num_slots=num_slots,
            slot_size=64,
            mlp_hidden_size=128
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x1, x2):
        # Extract features from both modalities
        feat1 = self.encoder1(x1)  # [batch_size, 64, patch_size, patch_size]
        feat2 = self.encoder2(x2)  # [batch_size, 64, patch_size, patch_size]
        
        # Combine features
        combined_feat = feat1 + feat2  # Simple addition fusion
        
        # Add position embedding
        x = self.pos_embed(combined_feat)
        
        # Flatten spatial dimensions for slot attention
        x = spatial_flatten(x)  # [batch_size, patch_size*patch_size, 64]
        
        # Apply layer norm and MLP
        x = self.mlp(self.layer_norm(x))
        
        # Slot attention
        slots = self.slot_attention(x)  # [batch_size, num_slots, 64]
        
        # Global average pooling over slots and classify
        slots = slots.transpose(1, 2)  # [batch_size, 64, num_slots]
        x = self.classifier(slots)
        
        return x


def build_grid(resolution):
    """Build coordinate grid for position embedding."""
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        super(SoftPositionEmbed, self).__init__()
        self.dense = nn.Linear(4, hidden_size, bias=True)
        grid = build_grid(resolution)
        self.register_buffer('grid', torch.from_numpy(grid))

    def forward(self, inputs):
        pos_emb = self.dense(self.grid)
        return inputs + pos_emb.unsqueeze(0).permute(0, 3, 1, 2)




