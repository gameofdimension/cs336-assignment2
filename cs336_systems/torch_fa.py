import math

import torch
from einops import einsum


def torch_fa_kernel(q, k, v, is_causal=False):
    Bq, Bk = 16, 16
    Tq, Tk = q.shape[-2] // Bq, k.shape[-2] // Bk
    d = q.shape[-1]
    output = torch.zeros_like(q)
    logsumexp = torch.zeros(q.shape[:-1], device=q.device)

    for i in range(Tq):
        q_tile = q[..., i * Bq:(i + 1) * Bq, :]
        o = torch.zeros_like(q_tile)
        denominator = torch.zeros(q_tile.shape[:-1], device=q_tile.device)
        max_score = torch.zeros(q_tile.shape[:-1], device=q_tile.device)
        torch.fill_(max_score, float('-inf'))
        for j in range(Tk):
            k_tile = k[..., j * Bk:(j + 1) * Bk, :]
            v_tile = v[..., j * Bk:(j + 1) * Bk, :]
            s = einsum(q_tile, k_tile, '... q d, ... k d -> ... q k') / math.sqrt(d)
            max_s = torch.max(s, dim=-1).values
            new_max_score = torch.max(max_score, max_s)
            p = torch.exp(s - new_max_score[..., None])
            adjustment = torch.exp(max_score - new_max_score)
            denominator = torch.sum(p, dim=-1) + denominator * adjustment

            incr = einsum(p, v_tile, '... q k, ... k d -> ... q d')
            o = einsum(torch.diag_embed(adjustment), o, '... b b, ... b d -> ... b d') + incr
            max_score = new_max_score
        o = o / denominator[..., None]
        output[..., i * Bq:(i + 1) * Bq, :] = o
        logsumexp[..., i * Bq:(i + 1) * Bq] = max_score + torch.log(denominator)
    return output, logsumexp


class TorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        ctx.is_causal = is_causal
        o, l = torch_fa_kernel(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, o, l)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "Backward pass is not implemented for TorchFlashAttention. Please implement it if needed.")
