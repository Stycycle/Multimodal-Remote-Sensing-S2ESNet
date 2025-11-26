import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. 基础模块 (保持不变)
# ==========================================

class Spatial_Enhance_Module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, size=None):
        """Implementation of SAEM: Spatial Enhancement Module"""
        super(Spatial_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

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
        batch_size = x1.size(0)
        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels, -1)
        t1 = t1.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)
        Affinity_M = self.dim_reduce(Affinity_M)
        Affinity_M = Affinity_M.view(batch_size, 1, x1.size(2), x1.size(3))

        x1 = x1 * Affinity_M.expand_as(x1)
        return x1


class Spectral_Enhance_Module(nn.Module):
    def __init__(self, in_channels, in_channels2, inter_channels=None, inter_channels2=None):
        """Implementation of SEEM: Spectral Enhancement Module"""
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

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels2, out_channels=self.inter_channels2, kernel_size=1),
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
        batch_size = x1.size(0)
        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels2, -1)
        t2 = t2.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)
        Affinity_M = self.dim_reduce(Affinity_M)
        Affinity_M = Affinity_M.view(batch_size, x1.size(1), 1, 1)

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
        return self.activation(self.bn(self.conv(x)))

# ==========================================
# 2. Slot Attention 相关组件 (辅助类)
# ==========================================

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
        # inputs: [B, C, H, W]
        pos_emb = self.dense(self.grid) # [H, W, C]
        pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
        return inputs + pos_emb

class SlotAttention(nn.Module):
    """Standard Slot Attention module."""
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

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_size))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_size))

        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(slot_size, slot_size, bias=False)
        self.project_v = nn.Linear(slot_size, slot_size, bias=False)

        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size)
        )

    def forward(self, inputs):
        # inputs: [batch_size, num_inputs (HxW), input_size (C)]
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        batch_size = inputs.size(0)
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
            batch_size, self.num_slots, self.slot_size, device=inputs.device)

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)
            q = q * (self.slot_size ** -0.5)
            
            attn_logits = torch.bmm(k, q.transpose(-2, -1))
            attn = F.softmax(attn_logits, dim=-1)

            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(-2, -1), v)

            slots = self.gru(updates.view(-1, self.slot_size), slots_prev.view(-1, self.slot_size))
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

# ==========================================
# 3. 融合核心：Slot-Based Gated Feature Fusion (重写)
# ==========================================

class GatedFusionModule(nn.Module):
    """
    对应图2中的黄色 "Slot-Based Gated Feature Fusion" 区域。
    将拼接后的Slots通过Tanh和Sigmoid门控机制进行融合。
    """
    def __init__(self, input_dim):
        super(GatedFusionModule, self).__init__()
        
        # 对应图中黄框内的 tanh 分支
        self.tanh_branch = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )
        
        # 对应图中黄框内的 sigma (sigmoid) 分支
        self.sigmoid_branch = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # 对应图中 MLP Pre-Head
        self.mlp_pre_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, slots):
        # slots: [B, 2*Num_Slots, Slot_Dim]
        
        # 1. 计算 Tanh(Content) 和 Sigmoid(Gate)
        content = self.tanh_branch(slots)
        gate = self.sigmoid_branch(slots)
        
        # 2. Element-wise Product (对应图中的 'x')
        gated_slots = content * gate
        
        # 3. Summation (对应图中的 '+')
        # 将所有 slots 的信息聚合为一个全局向量
        global_feat = torch.sum(gated_slots, dim=1) # [B, Slot_Dim]
        
        # 4. 右侧的 MLP Pre-Head 路径
        pre_head_out = self.mlp_pre_head(global_feat)
        
        # 5. 最终融合 (对应图中 MLP Pre-Head 上方的 'x' 符号，即 Modulation)
        # 这里的逻辑是：门控聚合后的特征 与 Pre-Head 处理后的特征相互作用
        # 也可以理解为残差，或者乘法调制。这里采用乘法融合(Modulation)以匹配 'x' 符号。
        final_feat = global_feat * pre_head_out 
        
        return final_feat


class SlotFusion(nn.Module):
    """
    根据新的流程图重构的 SlotFusion:
    SAEM/SEEM feats -> Object-Centric Decoupling (Parallel Slot Attn) -> Concat -> Gated Fusion -> MLP Head
    """
    def __init__(self, input_channels, input_channels2, n_classes, patch_size, num_slots=8, slot_dim=64, num_iterations=3):
        super(SlotFusion, self).__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        
        # 1. 特征投影与位置编码
        # 将输入的特征通道调整为 slot_dim
        self.conv1 = nn.Conv2d(input_channels, slot_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels2, slot_dim, kernel_size=1)
        
        self.pos_embed = SoftPositionEmbed(slot_dim, (patch_size, patch_size))
        self.layer_norm = nn.LayerNorm(slot_dim)
        
        self.mlp_mapper = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )

        # 2. Object-Centric Decoupling (Parallel Slot Attention)
        # 分别为两个模态建立 Slot Attention
        self.slot_attn_1 = SlotAttention(num_iterations, num_slots, slot_dim, slot_dim * 2)
        self.slot_attn_2 = SlotAttention(num_iterations, num_slots, slot_dim, slot_dim * 2)
        
        # 3. Slot-Based Gated Feature Fusion Module
        self.gated_fusion = GatedFusionModule(slot_dim)
        
        # 4. MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(slot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x1, x2):
        # x1: SAEM output [B, C1, H, W]
        # x2: SEEM output [B, C2, H, W]
        
        # --- Step 1: 预处理 (Project + PosEmbed + Flatten) ---
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        
        # 添加位置编码
        x1 = self.pos_embed(x1)
        x2 = self.pos_embed(x2)
        
        # 展平空间维度 [B, C, H, W] -> [B, H*W, C]
        x1 = x1.flatten(2).permute(0, 2, 1)
        x2 = x2.flatten(2).permute(0, 2, 1)
        
        # Layer Norm & MLP 映射
        x1 = self.mlp_mapper(self.layer_norm(x1))
        x2 = self.mlp_mapper(self.layer_norm(x2))
        
        # --- Step 2: 独立 Slot Attention (Object-Centric Decoupling) ---
        slots1 = self.slot_attn_1(x1) # [B, num_slots, slot_dim]
        slots2 = self.slot_attn_2(x2) # [B, num_slots, slot_dim]
        
        # --- Step 3: Concat ---
        # 拼接两个模态的slots
        slots_concat = torch.cat([slots1, slots2], dim=1) # [B, 2*num_slots, slot_dim]
        
        # --- Step 4: Gated Feature Fusion ---
        # 进入图中黄色的门控部分
        fused_feat = self.gated_fusion(slots_concat) # [B, slot_dim]
        
        # --- Step 5: MLP Head ---
        logits = self.mlp_head(fused_feat)
        
        return logits

# ==========================================
# 4. 主网络结构 (S2ESNet，已更新)
# ==========================================

class S2ESNet(nn.Module):

    def __init__(self, input_channels, input_channels2, n_classes, patch_size, num_slots=8, num_iterations=3):
        super(S2ESNet, self).__init__()

        # --- Forward - inverted CNN 部分 ---
        self.planes_a = [128, 64, 32]
        self.planes_b = [8, 16, 32]

        # 模态 A (e.g., HSI)
        self.conv1_a = conv_bn_relu(input_channels, self.planes_a[0], kernel_size=3, padding=1)
        self.conv2_a = conv_bn_relu(self.planes_a[0], self.planes_a[1], kernel_size=3, padding=1)
        self.conv3_a = conv_bn_relu(self.planes_a[1], self.planes_a[2], kernel_size=3, padding=1)

        # 模态 B (e.g., LiDAR/Audio)
        self.conv1_b = conv_bn_relu(input_channels2, self.planes_b[0], kernel_size=3, padding=1)
        self.conv2_b = conv_bn_relu(self.planes_b[0], self.planes_b[1], kernel_size=3, padding=1)
        self.conv3_b = conv_bn_relu(self.planes_b[1], self.planes_b[2], kernel_size=3, padding=1)

        # --- S2EM 部分 ---
        self.SAEM = Spatial_Enhance_Module(in_channels=self.planes_a[2], inter_channels=self.planes_a[2]//2, size=patch_size)
        self.SEEM = Spectral_Enhance_Module(in_channels=self.planes_b[2], in_channels2=self.planes_a[2])

        # --- 新的 SlotFusion 部分 (替代了原先的 Concat -> Conv -> Pool -> FC) ---
        # 这里的 inputs channels 直接对应 SAEM 和 SEEM 的输出通道数
        self.slot_fusion = SlotFusion(
            input_channels=self.planes_a[2],   # SAEM 输出通道
            input_channels2=self.planes_b[2],  # SEEM 输出通道
            n_classes=n_classes,
            patch_size=patch_size,
            num_slots=num_slots,
            num_iterations=num_iterations,
            slot_dim=64 # 可以自定义内部slot维度，通常64效果较好
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # 1. 特征提取 (Forward - inverted CNN)
        x1 = self.conv1_a(x1)
        x2 = self.conv1_b(x2)

        x1 = self.conv2_a(x1)
        x2 = self.conv2_b(x2)

        x1 = self.conv3_a(x1)
        x2 = self.conv3_b(x2)

        # 2. 空间/光谱增强 (S2EM)
        ss_x1 = self.SAEM(x1, x2) # 输出作为图2的上方输入分支
        ss_x2 = self.SEEM(x2, x1) # 输出作为图2的下方输入分支

        # 3. Slot-Based Gated Feature Fusion (图2的全流程)
        # 直接将增强后的特征传入新的 SlotFusion 模块
        x = self.slot_fusion(ss_x1, ss_x2)

        return x