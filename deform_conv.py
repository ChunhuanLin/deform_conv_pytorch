from torch.autograd import Variable
import torch
from torch import nn
import numpy as np


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.N = kernel_size**2
        self.padding = padding
        # Note: As illustrated in the paper, conv_offset's weights should be initialed with 0.
        self.conv_offset = nn.Conv2d(inc, 2*self.N, kernel_size=kernel_size, padding=padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        # (b, 2N, h, w)
        offset = self.conv_offset(x)
        dtype = offset.data.type()
        b, c, h, w = x.size()
        ks = self.kernel_size
        N = self.N
        zero = Variable(torch.FloatTensor([0]).type_as(offset.data), requires_grad=True)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(dtype)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(x.size(), dtype)

        p = (p_0 + p_n + offset).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # (b, 2N, h, w, 1, 1)
        p = p.expand(-1, -1, -1, -1, 1, 1)

        # (h, w)
        q = self._get_q(x.size(), dtype)

        # (b, N, h, w, h, w)
        G = torch.max((1-torch.abs(p[:, :N, :, :, :, :] - q[0, :, :])), zero)\
            * torch.max((1-torch.abs(p[:, N:, :, :, :, :] - q[1, :, :])), zero)
        # (b, N*h*w, h*w)
        G = G.contiguous().view(b, N*h*w, -1)
        # (b, h*w, c)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        # (b, c, h, w, N)
        x_offset = torch.bmm(G, x).contiguous().view(b, N, h, w, c).permute(0, 4, 2, 3, 1)
        # (b, c, h*kernel_size, w*kernel_size)
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1), order[x1, x2, ..., y1, y2, ...]
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*self.N, 1, 1))  # .repeat(b, axis=0).repeat(h, axis=2).repeat(w, axis=3)
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    def _get_p_0(self, x_size, dtype):
        b, c, h, w = x_size
        p_0_x, p_0_y = np.meshgrid(range(0, h), range(0, w), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(self.N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(self.N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_q(self, x_size, dtype):
        b, c, h, w = x_size
        q_x, q_y = np.meshgrid(range(0, h), range(0, w), indexing='ij')
        q_x = q_x.flatten().reshape(1, h, w)
        q_y = q_y.flatten().reshape(1, h, w)
        q = np.concatenate((q_x, q_y))
        q = Variable(torch.from_numpy(q).type(dtype), requires_grad=False)

        return q

    def _reshape_x_offset(self, x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
