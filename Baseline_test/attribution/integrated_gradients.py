import torch
import numpy as np

class IntegratedGradients_ECAPA:
    def __init__(self, model, n_steps=50):
        """
        针对ECAPA-TDNN的Integrated Gradients实现
        :param model: ECAPA_TDNN模型实例
        :param n_steps: 积分步数
        """
        self.model = model
        self.n_steps = n_steps

    def generate(self, input_tensor, baseline=None):
        """
        生成积分梯度归因
        :param input_tensor: 输入音频张量 [1, samples]
        :param baseline: 基线输入，默认为全零
        :return: 归因图 [80, T] (对应FBank特征)
        """
        self.model.eval()
        
        # 1. 获取FBank特征
        # 由于ECAPA_TDNN内部处理了FBank提取，我们需要手动提取以便在FBank层面进行归因
        # 注意：这里我们假设model.torchfbank是可用的
        with torch.no_grad():
            input_fbank = self.model.torchfbank(input_tensor) + 1e-6
            input_fbank = input_fbank.log()
            input_fbank = input_fbank - torch.mean(input_fbank, dim=-1, keepdim=True)
            # 此时input_fbank形状为 [1, 80, T]

        if baseline is None:
            baseline = torch.zeros_like(input_fbank)
            
        # 2. 生成插值路径
        # alphas: [n_steps+1, 1, 1, 1]
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=input_tensor.device).view(-1, 1, 1, 1)
        
        # interpolated_inputs: [n_steps+1, 80, T]
        interpolated_inputs = baseline + alphas * (input_fbank - baseline)
        interpolated_inputs.requires_grad_(True)
        
        # 3. 对插值输入进行前向传播
        # 由于我们绕过了torchfbank，需要定义一个从conv1开始的前向函数
        def forward_from_fbank(x):
            # 模拟ECAPA_TDNN.forward的后续部分
            # 注意：这里假设x已经经过了aug处理（在generate中我们不做aug）
            
            # 修复维度问题：
            # input_fbank: [1, 80, T]
            # interpolated_inputs: [n_steps+1, 1, 80, T] (因为alphas是[n+1, 1, 1, 1])
            # 但alphas * (input - baseline) 会产生 [n+1, 80, T]
            # 让我们检查 interpolated_inputs 的形状
            if x.dim() == 4 and x.shape[1] == 1:
                # 如果形状是 [B, 1, 80, T]，squeeze掉第1维
                x = x.squeeze(1)
            
            x = self.model.conv1(x)
            x = self.model.relu(x)
            x = self.model.bn1(x)

            x1 = self.model.layer1(x)
            x2 = self.model.layer2(x+x1)
            x3 = self.model.layer3(x+x1+x2)

            x = self.model.layer4(torch.cat((x1,x2,x3),dim=1))
            x = self.model.relu(x)

            t = x.size()[-1]

            global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
            
            w = self.model.attention(global_x)

            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

            x = torch.cat((mu,sg),1)
            x = self.model.bn5(x)
            x = self.model.fc6(x)
            x = self.model.bn6(x)
            return x

        # 为了避免显存OOM，我们可以分批处理插值输入
        # 这里简单起见，一次处理，如果步数不多应该没问题
        # 输出是embedding
        outputs = forward_from_fbank(interpolated_inputs)
        
        # 4. 计算梯度
        # 目标：最大化embedding的L2范数 (同Grad-CAM)
        score = outputs.norm(p=2, dim=1)
        
        grads = torch.autograd.grad(outputs=score, inputs=interpolated_inputs, 
                                   grad_outputs=torch.ones_like(score),
                                   create_graph=False, retain_graph=False)[0]
        
        # 5. 积分近似 (Trapezoidal rule)
        # avg_grads: [80, T]
        # 使用梯形法则：(y_0 + 2*y_1 + ... + 2*y_{n-1} + y_n) / (2*n)
        # 这里简化为平均值近似：mean(grads)
        avg_grads = torch.mean(grads, dim=0)
        
        # 6. 计算IG
        # (input - baseline) * avg_grads
        delta = (input_fbank - baseline).squeeze(0)
        ig = delta * avg_grads
        
        # 7. 验证收敛性 (可选)
        # with torch.no_grad():
        #     score_diff = forward_from_fbank(input_fbank).norm(p=2) - forward_from_fbank(baseline).norm(p=2)
        #     ig_sum = ig.sum()
        #     print(f"IG convergence delta: {abs(score_diff - ig_sum).item()}")

        return ig.cpu().detach().numpy()