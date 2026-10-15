
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class MultiScaleFeatureFusion(nn.Module):


    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=7, padding=3)
        self.fusion = nn.Conv1d((in_channels // 2) * 3, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return self.fusion(x)


class LowRankSparseAttention(nn.Module):

    
    def __init__(self, d_model, nhead, rank, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rank = rank
        self.dropout = nn.Dropout(dropout)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)


        self.U = nn.Parameter(torch.randn(self.head_dim, rank))
        self.V = nn.Parameter(torch.randn(self.head_dim, rank))

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)


        k_low = torch.matmul(k, self.U)
        q_low = torch.matmul(q, self.V)


        attn_scores = torch.matmul(q_low, k_low.transpose(-2, -1)) / (self.rank ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        return self.out_proj(output)


class TransformerEncoder(nn.Module):

    
    def __init__(self, d_model, nhead, rank, num_layers, dropout=0.1):
        super().__init__()

        self.freq_attention = FrequencyAttention(d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                LowRankSparseAttention(d_model, nhead, rank, dropout),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):

        x_freq = x.transpose(1, 2)
        x_freq = self.freq_attention(x_freq)
        x = x_freq.transpose(1, 2)


        for layer in self.layers:
            x = x + layer(x)
        return x


class HybridModel(nn.Module):

    
    def __init__(self, input_size, hidden_size=140, num_layers=4, num_classes=5, rank=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.msf = MultiScaleFeatureFusion(hidden_size * 2)
        self.transformer = TransformerEncoder(
            d_model=hidden_size * 2,
            nhead=4, rank=rank, num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        lstm_out, _ = self.lstm(x)


        fused_out = self.msf(lstm_out.transpose(1, 2)).transpose(1, 2)


        trans_out = self.transformer(fused_out)

        out = trans_out[:, -1, :]
        out = self.classifier(out)
        return out
