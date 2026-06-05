from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn
import numpy as np
import sys
import torch.nn.functional as F
import torch.fft as fft
import torch.nn.init as init

class USTKGQFLS(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int, int, int, int], rank: int, no_time_emb=False, no_location_emb=False, alpha: float = 10, init_size: float = 1e-2):
        super(USTKGQFLS, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=False)
            for s in [sizes[0], sizes[1], sizes[3], sizes[4]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.no_time_emb = no_time_emb
        self.no_location_emb = no_location_emb
        self.pi = 3.14159265358979323846
        self.weight_separation = 1.0
        self.weight_high_freq_intensity = 1.0
        self.alpha = alpha

    def decompose_relation_fft(self, rel_embedding):
        freq_domain = fft.fft(rel_embedding, dim=1)
        freqs = fft.fftfreq(self.rank, d=1.0).to(rel_embedding.device)
        abs_freqs = torch.abs(freqs)
        low_freq_mask = torch.exp(-abs_freqs * self.alpha).view(1, -1)
        high_freq_mask = 1.0 - low_freq_mask
        low_freq = freq_domain * low_freq_mask
        high_freq = freq_domain * high_freq_mask
        low_freq_component = fft.ifft(low_freq, dim=1).real
        high_freq_component = fft.ifft(high_freq, dim=1).real

        return low_freq_component, high_freq_component, low_freq, high_freq

    def loss_freq(self, low_freq, high_freq):
        separation_loss = - torch.norm(low_freq - high_freq, p=2)
        high_freq_intensity_loss = torch.norm(high_freq, p=2)
        total_loss = (self.weight_separation * separation_loss \
                    + self.weight_high_freq_intensity * high_freq_intensity_loss) * (1 / (high_freq.shape[0] * high_freq.shape[0]))
        return total_loss

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        location = self.embeddings[3](x[:, 4])
        fuzzy = (x[:, 5]/1000000).unsqueeze(1)

        lhs = self.fuzzy_gate(lhs, fuzzy)
        rel = self.fuzzy_gate(rel, fuzzy)
        time = self.fuzzy_gate(time, fuzzy)
        location = self.fuzzy_gate(location, fuzzy)

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]
        location = location[:, :self.rank], location[:, self.rank:]

        rel_low, rel_high, _, _ = self.decompose_relation_fft(rel[0])

        time_smoothed = time[0].mean(dim=1, keepdim=True)
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])
        rt_low = (rel_low + time_smoothed) * time[1]
        rt_high = (rel_high + time_gradient) * time[1]

        location_smoothed = location[0].mean(dim=1, keepdim=True)
        location_gradient = torch.diff(location[0], dim=1, prepend=location[0][:, 0:1])
        rl_low = (rel_low + location_smoothed) * location[1]
        rl_high = (rel_high + location_gradient) * location[1]

        rtl = 0.25 * rt_low + 0.25 * rt_high + 0.25 * rl_low + 0.25 * rl_high, rel[1]

        return torch.sum(((lhs[0] + rtl[1]) * rtl[0]) * rhs[0], 1, keepdim=True)

    def fuzzy_gate(self, element, fuzzy):
        gate = nn.Sigmoid()(element.mean(-1, keepdim=True))
        fuzzy_vector = fuzzy * element + (1 - fuzzy) * element * gate
        return fuzzy_vector

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        location = self.embeddings[3](x[:, 4])
        fuzzy = (x[:, 5]/1000000).unsqueeze(1)

        lhs = self.fuzzy_gate(lhs, fuzzy)
        rel = self.fuzzy_gate(rel, fuzzy)
        time = self.fuzzy_gate(time, fuzzy)
        location = self.fuzzy_gate(location, fuzzy)

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]
        location = location[:, :self.rank], location[:, self.rank:]

        rel_low, rel_high, low_freq, high_freq = self.decompose_relation_fft(rel[0])

        time_smoothed = time[0].mean(dim=1, keepdim=True)
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])
        rt_low = (rel_low + time_smoothed) * time[1]
        rt_high = (rel_high + time_gradient) * time[1]

        location_smoothed = location[0].mean(dim=1, keepdim=True)
        location_gradient = torch.diff(location[0], dim=1, prepend=location[0][:, 0:1])
        rl_low = (rel_low + location_smoothed) * location[1]
        rl_high = (rel_high + location_gradient) * location[1]

        rtl = 0.25 * rt_low + 0.25 * rt_high + 0.25 * rl_low + 0.25 * rl_high, rel[1]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        loss_freq = self.loss_freq(low_freq, high_freq)

        return ((
            ((lhs[0] + rtl[1]) * rtl[0]) @ right[0].t()
        ), (
            torch.sqrt(lhs[0] ** 2),
            torch.sqrt(rtl[0] ** 2 + rtl[1] ** 2),
            torch.sqrt(rhs[0] ** 2)
        ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight,
                self.embeddings[3].weight[:-1] if self.no_location_emb else self.embeddings[3].weight, loss_freq)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size][:, :self.rank].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1]) 
        time = self.embeddings[2](queries[:, 3])
        location = self.embeddings[3](queries[:, 4])
        fuzzy = (queries[:, 5]/1000000).unsqueeze(1)

        lhs = self.fuzzy_gate(lhs, fuzzy)
        rel = self.fuzzy_gate(rel, fuzzy)
        time = self.fuzzy_gate(time, fuzzy)
        location = self.fuzzy_gate(location, fuzzy)

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]
        location = location[:, :self.rank], location[:, self.rank:]

        rel_low, rel_high, _, _ = self.decompose_relation_fft(rel[0])

        time_smoothed = time[0].mean(dim=1, keepdim=True)
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])
        rt_low = (rel_low + time_smoothed) * time[1]
        rt_high = (rel_high + time_gradient) * time[1]

        location_smoothed = location[0].mean(dim=1, keepdim=True)
        location_gradient = torch.diff(location[0], dim=1, prepend=location[0][:, 0:1])
        rl_low = (rel_low + location_smoothed) * location[1]
        rl_high = (rel_high + location_gradient) * location[1]

        rtl = 0.25 * rt_low + 0.25 * rt_high + 0.25 * rl_low + 0.25 * rl_high, rel[1]

        return (lhs[0] + rtl[1]) * rtl[0]