import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm  # 必须安装: pip install timm

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

# ==========================================
# 2. 新增模块: Swin Transformer 特征提取 (替换 conv1/2/3)
# ==========================================

class SwinFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinFeatureExtractor, self).__init__()
        # 使用轻量级的 swin_tiny 作为 backbone
        # features_only=True 表示只取中间层的特征图
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=False, 
            in_chans=in_channels, 
            features_only=True,
            img_size=(32,32)
        )
        # Swin Tiny Stage 0 的输出通道通常是 96
        # 我们用 1x1 卷积将其映射到 32 通道，以匹配后续模块
        self.adapter = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 记录原始尺寸 (H, W)
        H, W = x.shape[2], x.shape[3]
        
        # Swin 需要最小 32x32 的输入，如果 patch_size 太小需要放大
        if H < 32 or W < 32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        
        # 提取特征 (取第一层的输出)
        feats = self.backbone(x)
        x_feat = feats[0] # [B, 96, H/4, W/4]

        # 2. 核心修复：检查并转换维度顺序
        # 如果最后一维是 96，说明是 [B, H, W, C] 格式，需要转为 [B, C, H, W]
        if x_feat.shape[-1] == 96:
            x_feat = x_feat.permute(0, 3, 1, 2).contiguous()
        
        # 通道调整
        x_feat = self.adapter(x_feat)
        
        # 将空间尺寸还原回原始输入的 patch_size
        if x_feat.shape[2:] != (H, W):
            x_feat = F.interpolate(x_feat, size=(H, W), mode='bilinear', align_corners=False)
            
        return x_feat

# ==========================================
# 3. Slot Attention 组件 (保持不变)
# ==========================================

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    return torch.from_numpy(grid.astype(np.float32)).unsqueeze(0) # [1, H, W, 2]

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super(SoftPositionEmbed, self).__init__()
        self.dense = nn.Linear(4, hidden_size)
        grid = build_grid(resolution)
        # 修改 grid 逻辑以支持 4 通道坐标拼接 (grid 和 1-grid)
        grid_full = torch.cat([grid, 1.0 - grid], dim=-1)
        self.register_buffer('grid', grid_full)

    def forward(self, inputs):
        pos_emb = self.dense(self.grid).permute(0, 3, 1, 2)
        return inputs + pos_emb

class SlotAttention(nn.Module):
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

class GatedFusionModule(nn.Module):
    def __init__(self, input_dim):
        super(GatedFusionModule, self).__init__()
        self.tanh_branch = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh())
        self.sigmoid_branch = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.mlp_pre_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, slots):
        content = self.tanh_branch(slots)
        gate = self.sigmoid_branch(slots)
        gated_slots = content * gate
        global_feat = torch.sum(gated_slots, dim=1)
        pre_head_out = self.mlp_pre_head(global_feat)
        return global_feat * pre_head_out

class SlotFusion(nn.Module):
    def __init__(self, input_channels, input_channels2, n_classes, patch_size, num_slots=8, slot_dim=64, num_iterations=3):
        super(SlotFusion, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, slot_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels2, slot_dim, kernel_size=1)
        self.pos_embed = SoftPositionEmbed(slot_dim, (patch_size, patch_size))
        self.layer_norm = nn.LayerNorm(slot_dim)
        self.mlp_mapper = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )
        self.slot_attn_1 = SlotAttention(num_iterations, num_slots, slot_dim, slot_dim * 2)
        self.slot_attn_2 = SlotAttention(num_iterations, num_slots, slot_dim, slot_dim * 2)
        self.gated_fusion = GatedFusionModule(slot_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(slot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x1, x2):
        x1 = self.pos_embed(self.conv1(x1)).flatten(2).permute(0, 2, 1)
        x2 = self.pos_embed(self.conv2(x2)).flatten(2).permute(0, 2, 1)
        x1 = self.mlp_mapper(self.layer_norm(x1))
        x2 = self.mlp_mapper(self.layer_norm(x2))
        slots1 = self.slot_attn_1(x1)
        slots2 = self.slot_attn_2(x2)
        fused_feat = self.gated_fusion(torch.cat([slots1, slots2], dim=1))
        return self.mlp_head(fused_feat)

# ==========================================
# 4. 主网络结构 (S2ESNet - Swin Transformer 版本)
# ==========================================

class S2ESNet_Swin(nn.Module):
    def __init__(self, input_channels, input_channels2, n_classes, patch_size, num_slots=8, num_iterations=3):
        super(S2ESNet_Swin, self).__init__()

        # 固定目标特征通道数为 32，以适配 SAEM 和 SEEM
        self.target_channels = 32

        # --- 替换原来的 conv1/2/3 为 SwinTransformer ---
        print("Initializing S2ESNet with Swin Transformer backbones...")
        self.extractor_a = SwinFeatureExtractor(input_channels, self.target_channels)
        self.extractor_b = SwinFeatureExtractor(input_channels2, self.target_channels)

        # --- S2EM 部分 (通道数均设为 32) ---
        self.SAEM = Spatial_Enhance_Module(in_channels=self.target_channels, inter_channels=self.target_channels//2, size=patch_size)
        self.SEEM = Spectral_Enhance_Module(in_channels=self.target_channels, in_channels2=self.target_channels)

        # --- SlotFusion 部分 ---
        self.slot_fusion = SlotFusion(
            input_channels=self.target_channels,   
            input_channels2=self.target_channels,  
            n_classes=n_classes,
            patch_size=patch_size,
            num_slots=num_slots,
            num_iterations=num_iterations,
            slot_dim=64
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # 1. Swin Transformer 特征提取
        x1 = self.extractor_a(x1) # [B, 32, H, W]
        x2 = self.extractor_b(x2) # [B, 32, H, W]

        # 2. 空间/光谱增强 (S2EM)
        ss_x1 = self.SAEM(x1, x2) 
        ss_x2 = self.SEEM(x2, x1) 

        # 3. Slot-Based 融合
        x = self.slot_fusion(ss_x1, ss_x2)

        return x