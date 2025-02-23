import torch
from backbones.vit import VisionTransformer

vit = VisionTransformer(img_size=112, patch_size=9, num_classes=512, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1).cuda()
weight = torch.load('/home/quocnc1/Documents/wf4m_vit_s_rgb_result/model.pt')
vit.load_state_dict(weight)
vit.eval()

xx = torch.randn(1, 3, 112, 112).cuda()
traced_cell = torch.jit.trace(vit, (xx))
torch.jit.save(traced_cell, 'wf4m_vit_s_rgb_result.pt')