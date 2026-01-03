import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from safetensors.torch import load_file

# 改进点：改为残差结构 (Identity + Attention)
class Residual_Spatial_Enhance_Module(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(Residual_Spatial_Enhance_Module, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        
        # 1x1 卷积用于计算注意力图
        self.conv_q = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.conv_k = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1) # Value 保持通道数
        
        # 输出融合层
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.gamma = nn.Parameter(torch.zeros(1)) # 学习一个加权系数，初始为0

    def forward(self, x_main, x_guide):
        # x_main: 需要增强的模态 (如 HSI)
        # x_guide:以此为参考的模态 (如 LiDAR)
        B, C, H, W = x_main.shape
        
        # 计算空间注意力 map
        Q = self.conv_q(x_main).view(B, -1, H*W).permute(0, 2, 1) # [B, N, C']
        K = self.conv_k(x_guide).view(B, -1, H*W)                # [B, C', N]
        
        # Attention Matrix
        attn = torch.bmm(Q, K) # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        
        V = self.conv_v(x_guide).view(B, -1, H*W) # [B, C, N]
        
        # 聚合特征
        out = torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)
        out = self.out_conv(out)
        
        # 残差连接：原始特征 + 加权后的增强特征
        return x_main + self.gamma * out

class Residual_Spectral_Enhance_Module(nn.Module):
    def __init__(self, in_channels):
        super(Residual_Spectral_Enhance_Module, self).__init__()
        # SEEM 通常用于通道注意力 (Channel Attention)
        # 改进：使用类似 SE-Block 或 ECA 的机制，利用另一模态生成通道权重
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_main, x_guide):
        B, C, _, _ = x_main.shape
        # 全局池化获取光谱/通道统计量
        y_main = self.avg_pool(x_main).view(B, C)
        y_guide = self.avg_pool(x_guide).view(B, C)
        
        # 拼接两个模态的统计量来生成门控权重
        y_cat = torch.cat([y_main, y_guide], dim=1)
        
        # 生成通道权重 [B, C]
        attn = self.fc(y_cat).view(B, C, 1, 1)
        
        # 也是残差连接： x * attn + x
        return x_main * attn + x_main
    
class ImprovedSwinAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, checkpoint_path=None):
        super(ImprovedSwinAdapter, self).__init__()
        
        # 1. 创建模型 (保持 pretrained=False)
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=False, 
            features_only=True,
            in_chans=in_channels, 
            img_size=(32, 32)
        )

        # 2. 【核心步骤】加载并转换 safetensors 权重
        if checkpoint_path and checkpoint_path.endswith('.safetensors'):
            try:
                # 使用 safetensors 专属方法加载
                state_dict = load_file(checkpoint_path)
                model_dict = self.backbone.state_dict()
                new_state_dict = {}

                print(f"Loading weights from {checkpoint_path}...")

                for k, v in state_dict.items():
                    # --- 转换逻辑 ---
                    # 预训练权重里的 key 是 "layers.0.blocks..."
                    # timm features_only 后的 key 是 "layers_0.blocks..." (点变下划线)
                    nk = k.replace('layers.', 'layers_')
                    
                    if nk in model_dict:
                        # 检查形状是否匹配 (会自动跳过 3通道 vs 180通道的 patch_embed)
                        if v.shape == model_dict[nk].shape:
                            new_state_dict[nk] = v
                        else:
                            print(f"  [Skip] Shape mismatch for {nk}: {v.shape} -> {model_dict[nk].shape}")
                    else:
                        # 兜底：有些 key 可能不需要转换 (如 norm)
                        if k in model_dict and v.shape == model_dict[k].shape:
                            new_state_dict[k] = v

                # 加载权重
                msg = self.backbone.load_state_dict(new_state_dict, strict=False)
                print(f"Successfully matched {len(new_state_dict)} tensors.")
                print(f"Missing keys (expected for HSI): {len(msg.missing_keys)}")
            except Exception as e:
                print(f"Failed to load safetensors: {e}")

        # 后续 Adapter 层
        self.adapter0 = nn.Conv2d(96, out_channels, 1)
        self.adapter1 = nn.Conv2d(192, out_channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 确保输入不小于 32x32
        H, W = x.shape[2], x.shape[3]
        if H < 32 or W < 32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
            
        feats = self.backbone(x)
        
        # 特征 0: [B, 96, 8, 8] (假设输入32x32)
        f0 = feats[0].permute(0, 3, 1, 2).contiguous()
        
        # 特征 1: [B, 192, 4, 4]
        f1 = feats[1].permute(0, 3, 1, 2).contiguous()
        
        # 调整通道
        f0 = self.adapter0(f0)
        f1 = self.adapter1(f1)
        
        # 将 f1 上采样与 f0 对齐
        f1_up = F.interpolate(f1, size=f0.shape[2:], mode='bilinear', align_corners=False)
        
        # 拼接融合
        out = torch.cat([f0, f1_up], dim=1)
        out = self.fuse(out) # [B, 32, 8, 8]
        
        # 恢复到原始输入尺寸 (如果需要)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            
        return out
    
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

class JointSlotFusion(nn.Module):
    def __init__(self, input_dim, n_classes, num_slots=8, slot_dim=64, num_iterations=3):
        super(JointSlotFusion, self).__init__()
        
        # 将两个模态的输入投影到同一维度
        self.proj_x1 = nn.Conv2d(input_dim, slot_dim, 1)
        self.proj_x2 = nn.Conv2d(input_dim, slot_dim, 1)
        
        self.pos_embed = SoftPositionEmbed(slot_dim, (32, 32)) # 假设输入已resize或adapt
        
        # 只需要一个 Slot Attention 模块
        self.slot_attn = SlotAttention(num_iterations, num_slots, slot_dim, slot_dim * 2)
        
        self.norm = nn.LayerNorm(slot_dim)
        
        # 分类头：使用 Attention Pooling 而不是简单的 Sum
        self.class_token = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.head_norm = nn.LayerNorm(slot_dim)
        self.head = nn.Linear(slot_dim, n_classes)

    def forward(self, x1, x2):
        # 1. 投影并添加位置编码
        # 注意：这里需要确保 x1, x2 尺寸一致
        x1 = self.proj_x1(x1) 
        x2 = self.proj_x2(x2)
        
        # 为了简单，这里直接使用加法或者拼接。
        # 更好的做法是将两者在序列维度拼接：Sequence Concat
        
        # 展平: [B, C, H, W] -> [B, N, C]
        x1_flat = x1.flatten(2).permute(0, 2, 1)
        x2_flat = x2.flatten(2).permute(0, 2, 1)
        
        # 关键策略：序列拼接。Slots 将同时看到模态 A 和 模态 B 的像素
        # inputs shape: [B, H*W*2, slot_dim]
        inputs = torch.cat([x1_flat, x2_flat], dim=1)
        
        inputs = self.norm(inputs)
        
        # 2. 运行 Slot Attention
        # Slots 会自动学习：有的 slot 关注 HSI 的光谱，有的 slot 关注 LiDAR 的高程
        slots = self.slot_attn(inputs) # [B, num_slots, slot_dim]
        
        # 3. 分类输出
        # 简单策略：取 Mean 或者 Max
        # 进阶策略：使用类似 ViT 的 Class Token 机制，这里用 Mean 最稳健
        fused_feat = torch.mean(slots, dim=1) 
        
        logits = self.head(fused_feat)
        return logits
    
class S2ESNet_Pro(nn.Module):
    def __init__(self, input_channels, input_channels2, n_classes, patch_size, num_slots=8):
        super(S2ESNet_Pro, self).__init__()
        
        self.target_channels = 32

        # 1. 改进的骨干网 (Pretrained + Multi-scale)
        print("Initializing Improved S2ESNet...")
        self.extractor_a = ImprovedSwinAdapter(input_channels, self.target_channels)
        self.extractor_b = ImprovedSwinAdapter(input_channels2, self.target_channels)

        # 2. 改进的残差增强模块
        self.SAEM = Residual_Spatial_Enhance_Module(self.target_channels)
        self.SEEM = Residual_Spectral_Enhance_Module(self.target_channels)

        # 3. 改进的联合 Slot Fusion
        self.fusion = JointSlotFusion(
            input_dim=self.target_channels,
            n_classes=n_classes,
            num_slots=num_slots,
            slot_dim=64
        )

    def forward(self, x1, x2):
        # Backbone
        f1 = self.extractor_a(x1)
        f2 = self.extractor_b(x2)
        
        # Interaction (Residual)
        # x1 结合 x2 的信息增强
        f1_enhanced = self.SAEM(f1, f2)
        # x2 结合 x1 的信息增强
        f2_enhanced = self.SEEM(f2, f1)
        
        # Joint Fusion
        logits = self.fusion(f1_enhanced, f2_enhanced)
        
        return logits