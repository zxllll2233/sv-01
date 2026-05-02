import torch
import torch.nn.functional as F
import numpy as np

class GradCAM_ECAPA:
    def __init__(self, model, target_layer_name="layer4"):
        """
        针对ECAPA-TDNN的Grad-CAM实现
        :param model: ECAPA_TDNN模型实例
        :param target_layer_name: 目标层名称 (默认'layer4'，即Conv1d(3*C, 1536, 1))
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # 根据名称查找目标层
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            # 尝试直接从属性获取
            if hasattr(self.model, self.target_layer_name):
                target_layer = getattr(self.model, self.target_layer_name)

        if target_layer is None:
            raise ValueError(f"Target layer {self.target_layer_name} not found in model")

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, target_index=None):
        """
        生成Grad-CAM热力图
        :param input_tensor: 输入音频张量 [1, samples]
        :param target_index: 目标类别索引 (对于声纹识别embedding，这里我们最大化embedding的L2范数或特定维度的值)
                            由于这是无监督/自监督或metric learning，我们通常最大化embedding的模长作为"重要性"的代理
        :return: 热力图 [T]
        """
        self.model.eval()
        self.model.zero_grad()
        
        # 前向传播
        # 注意：ECAPA_TDNN.forward(x, aug)
        output = self.model(input_tensor, aug=False)
        
        # 定义目标：这里我们最大化embedding的L2范数，因为重要的特征应该贡献更大的模长
        # 或者如果有特定的speaker loss，可以用loss对输入的梯度，但这里我们在推断模式下
        # 简化起见，我们对输出embedding求和作为标量进行反向传播
        score = output.norm(p=2, dim=1)
        
        score.backward()
        
        # 获取梯度和激活
        gradients = self.gradients # [B, C, T]
        activations = self.activations # [B, C, T]
        
        if gradients is None or activations is None:
            return None
            
        # 全局平均池化梯度作为权重
        weights = torch.mean(gradients, dim=2, keepdim=True) # [B, C, 1]
        
        # 加权激活
        cam = torch.sum(weights * activations, dim=1) # [B, T]
        
        # ReLU
        cam = F.relu(cam)
        
        # 归一化
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.squeeze().cpu().numpy()

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []