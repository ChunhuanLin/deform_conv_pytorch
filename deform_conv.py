from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        # Note: As illustrated in the paper, conv_offset's weights should be initialed with 0.
        self.conv_offset = nn.Conv2d(inc, 2*self.kernel_size**2, kernel_size=kernel_size, padding=padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        # (b, 2N, h, w)
        offset = self.conv_offset(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        q_lt = p.floor().long()
        q_rb = p.ceil().long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        overlap_indices = torch.eq(q_lt, q_rb) + 1
        overlap = (overlap_indices[..., :N] * overlap_indices[..., N:]).float()

        # bilinear kernel (b, h, w, N)
        g_lt_x = 1 - torch.abs(p[..., :N] - q_lt[..., :N].type_as(p))
        g_lt_y = 1 - torch.abs(p[..., N:] - q_lt[..., N:].type_as(p))
        g_rb_x = 1 - torch.abs(p[..., :N] - q_rb[..., :N].type_as(p))
        g_rb_y = 1 - torch.abs(p[..., N:] - q_rb[..., N:].type_as(p))
        g_lb_x = 1 - torch.abs(p[..., :N] - q_lb[..., :N].type_as(p))
        g_lb_y = 1 - torch.abs(p[..., N:] - q_lb[..., N:].type_as(p))
        g_rt_x = 1 - torch.abs(p[..., :N] - q_rt[..., :N].type_as(p))
        g_rt_y = 1 - torch.abs(p[..., N:] - q_rt[..., N:].type_as(p))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = (g_lt_x * g_lt_y).unsqueeze(dim=1) * x_q_lt + \
                 (g_rb_x * g_rb_y).unsqueeze(dim=1) * x_q_rb + \
                 (g_lb_x * g_lb_y).unsqueeze(dim=1) * x_q_lb + \
                 (g_rt_x * g_rt_y).unsqueeze(dim=1) * x_q_rt

        x_offset /= overlap.unsqueeze(dim=1)

        # (b, c, h*kernel_size, w*kernel_size)
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1), order[x1, x2, ..., y1, y2, ...]
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(0, h), range(0, w), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        p = torch.cat([torch.clamp(p[:, :N, :, :], 0, h-1), torch.clamp(p[:, N:, :, :], 0, w-1)], dim=1)

        return p

    @staticmethod
    def _get_q(x_size, dtype):
        b, c, h, w = x_size
        q_x, q_y = np.meshgrid(range(0, h), range(0, w), indexing='ij')
        q_x = q_x.flatten().reshape(1, h, w)
        q_y = q_y.flatten().reshape(1, h, w)
        q = np.concatenate((q_x, q_y))
        q = Variable(torch.from_numpy(q).type(dtype), requires_grad=False)

        return q

    def _get_x_q(self, x, q, N):
        b, c, h, w = x.size()
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*w + q[..., N:]  # offset_x*offset_y + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
