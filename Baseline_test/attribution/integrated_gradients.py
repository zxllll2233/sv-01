import torch
import torch.nn.functional as F
import numpy as np


class IntegratedGradients_ECAPA:
    def __init__(self, model, n_steps=50, convergence_threshold=0.1):
        """
        针对ECAPA-TDNN的Integrated Gradients实现
        
        Args:
            model: ECAPA_TDNN模型实例
            n_steps: 积分步数
            convergence_threshold: IG收敛性验证阈值（相对误差）
        """
        self.model = model
        self.n_steps = n_steps
        self.convergence_threshold = convergence_threshold

    def _extract_fbank(self, audio_tensor):
        """从原始音频提取FBank特征 [1, 80, T]"""
        with torch.no_grad():
            fbank = self.model.torchfbank(audio_tensor) + 1e-6
            fbank = fbank.log()
            fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
        return fbank

    def _forward_from_fbank(self, x):
        """从FBank特征开始前向传播，返回embedding"""
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = self.model.conv1(x)
        x = self.model.relu(x)
        x = self.model.bn1(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x + x1)
        x3 = self.model.layer3(x + x1 + x2)

        x = self.model.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.model.relu(x)

        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)

        w = self.model.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.model.bn5(x)
        x = self.model.fc6(x)
        x = self.model.bn6(x)
        return x

    def generate(self, input_tensor, ref_tensor=None, baseline=None,
                 objective='cosine_sim', verify_convergence=True):
        """
        生成积分梯度归因
        
        Args:
            input_tensor: 输入音频张量 [1, samples]
            ref_tensor: 参考音频张量 [1, samples]（cosine_sim 目标时必需）
            baseline: 基线FBank张量 [1, 80, T] 或 None（使用zero baseline）
            objective: 归因目标
                'cosine_sim' - 最大化与参考embedding的余弦相似度（推荐）
                'l2_norm' - 最大化embedding L2范数（旧版兼容）
            verify_convergence: 是否验证IG收敛性
            
        Returns:
            ig: 归因图 [80, T] (numpy)
        """
        self.model.eval()

        # 1. 提取输入FBank
        input_fbank = self._extract_fbank(input_tensor)  # [1, 80, T]

        # 2. 设置基线
        if baseline is None:
            baseline = torch.zeros_like(input_fbank)
        else:
            # 确保baseline与input_fbank形状一致
            if baseline.shape != input_fbank.shape:
                # baseline可能是 [80, 1]，需要广播到 [1, 80, T]
                baseline = baseline.expand_as(input_fbank)

        # 3. 计算参考embedding（如果使用cosine_sim目标）
        ref_embedding = None
        if objective == 'cosine_sim':
            if ref_tensor is None:
                raise ValueError("cosine_sim objective requires ref_tensor")
            with torch.no_grad():
                ref_fbank = self._extract_fbank(ref_tensor)
                ref_embedding = self._forward_from_fbank(ref_fbank)
                # 归一化参考embedding并detach，梯度不流向参考
                ref_embedding = F.normalize(ref_embedding, p=2, dim=1).detach()

        # 4. 分批计算梯度（避免OOM：不一次性处理所有插值步骤）
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=input_tensor.device)
        delta = input_fbank - baseline  # [1, 80, T]

        all_grads = []
        batch_size = max(1, min(10, self.n_steps + 1))

        for batch_start in range(0, self.n_steps + 1, batch_size):
            batch_end = min(batch_start + batch_size, self.n_steps + 1)
            batch_alphas = alphas[batch_start:batch_end].view(-1, 1, 1, 1)
            batch_inputs = baseline + batch_alphas * delta
            batch_inputs.requires_grad_(True)

            batch_outputs = self._forward_from_fbank(batch_inputs)

            if objective == 'cosine_sim':
                assert ref_embedding is not None
                batch_outputs_norm = F.normalize(batch_outputs, p=2, dim=1)
                batch_score = torch.sum(batch_outputs_norm * ref_embedding, dim=1)
            else:
                batch_score = batch_outputs.norm(p=2, dim=1)

            batch_grads = torch.autograd.grad(
                outputs=batch_score,
                inputs=batch_inputs,
                grad_outputs=torch.ones_like(batch_score),
                create_graph=False,
                retain_graph=False
            )[0]

            all_grads.append(batch_grads.detach())
            del batch_inputs, batch_outputs, batch_score, batch_grads

        grads = torch.cat(all_grads, dim=0)  # [n_steps+1, 80, T]
        del all_grads

        # 8. 梯形法则积分近似
        # (y_0 + 2*y_1 + ... + 2*y_{n-1} + y_n) / (2*n)
        grads_arr = grads  # [n_steps+1, 80, T]
        avg_grads = (grads_arr[0] + 2 * grads_arr[1:-1].sum(dim=0) + grads_arr[-1]) / (2 * self.n_steps)

        # 9. 计算IG
        delta = (input_fbank - baseline).squeeze(0)  # [80, T]
        ig = delta * avg_grads  # [80, T]

        # 10. 收敛性验证
        if verify_convergence:
            with torch.no_grad():
                f_input = self._forward_from_fbank(input_fbank)
                f_baseline = self._forward_from_fbank(baseline)

                if objective == 'cosine_sim':
                    assert ref_embedding is not None
                    f_input_norm = F.normalize(f_input, p=2, dim=1)
                    f_baseline_norm = F.normalize(f_baseline, p=2, dim=1)
                    score_diff = (torch.sum(f_input_norm * ref_embedding, dim=1) -
                                  torch.sum(f_baseline_norm * ref_embedding, dim=1))
                else:
                    score_diff = f_input.norm(p=2, dim=1) - f_baseline.norm(p=2, dim=1)

                ig_sum = ig.sum()
                relative_error = abs((score_diff - ig_sum).item()) / (abs(score_diff).item() + 1e-8)

                if relative_error > self.convergence_threshold:
                    print(f"[IG Warning] Convergence check: relative_error={relative_error:.4f} "
                          f"(threshold={self.convergence_threshold}). "
                          f"Consider increasing n_steps (current={self.n_steps})")
                else:
                    print(f"[IG] Convergence check passed: relative_error={relative_error:.4f}")

        return ig.cpu().detach().numpy()
