"""
通用工具模块：包含Transformer核心组件、训练工具、可视化函数。
所有注释与输出均为中文，便于多份Notebook复用。
"""
import math
import copy
import warnings
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings("ignore")

# 设置随机种子和设备
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 核心组件 =====
class EncoderDecoder(nn.Module):
    """
    标准的Encoder-Decoder架构：先编码源序列，再解码目标序列。
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    线性层 + log softmax，将decoder输出映射到词表概率。
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    深拷贝N个相同层，便于构建堆叠结构。
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """
    对最后一维做归一化，保持数值稳定。
    """

    def __init__(self, features, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    残差 + LayerNorm + Dropout。
    顺序采用Post-LN：先Norm再子层。
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer: Callable[[torch.Tensor], torch.Tensor]):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    """由N个EncoderLayer堆叠，并在末尾做LayerNorm。"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    单层Encoder：Self-Attention + Feed Forward。
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """由N个DecoderLayer堆叠，并在末尾做LayerNorm。"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    单层Decoder：Masked Self-Attn + Cross-Attn + Feed Forward。
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size: int):
    """
    Decoder用下三角mask，避免看到未来位置。
    返回形状：(1, size, size)。
    """

    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent == 0


def attention(query, key, value, mask: Optional[torch.Tensor] = None, dropout=None):
    """
    Scaled Dot-Product Attention。
    返回 (输出, 注意力权重)。
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention，将d_model分成h个头并行计算。
    """

    def __init__(self, h, d_model, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """逐位置两层前馈网络。"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    """Embedding层，乘以sqrt(d_model)做尺度匹配。"""

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    正弦位置编码，注入绝对位置信息。
    """

    def __init__(self, d_model, dropout, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
    src_vocab, tgt_vocab, N: int = 6, d_model: int = 512, d_ff: int = 2048, h: int = 8, dropout: float = 0.1
):
    """
    根据超参数构建完整Transformer并做Xavier初始化。
    """

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ===== 可视化工具 =====
def visualize_mask(size: int = 20):
    """可视化decoder下三角mask。"""

    mask = subsequent_mask(size)
    plt.figure(figsize=(8, 8))
    plt.imshow(mask[0].cpu().numpy(), cmap="YlGn", interpolation="nearest")
    plt.title("Subsequent Mask\n(Yellow=Visible, Green=Masked)", fontsize=14)
    plt.xlabel("Positions that current position can attend to (Column)")
    plt.ylabel("Current Position (Row)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def visualize_positional_encoding(d_model: int = 20, max_len: int = 100):
    """可视化位置编码正弦波形与热力图。"""

    pe = PositionalEncoding(d_model, dropout=0)
    y = pe.forward(torch.zeros(1, max_len, d_model))

    plt.figure(figsize=(15, 5))
    dims_to_show = [4, 5, 6, 7]
    for dim in dims_to_show:
        plt.plot(y[0, :, dim].numpy(), label=f"Dimension {dim}")
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.title("Sinusoidal Pattern of Positional Encoding\nDifferent dimensions have different frequencies")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.imshow(y[0].numpy().T, aspect="auto", cmap="RdBu", interpolation="nearest")
    plt.colorbar(label="Encoding Value")
    plt.xlabel("Position")
    plt.ylabel("Dimension")
    plt.title("Positional Encoding Heatmap\nEach column represents the encoding vector for one position")
    plt.tight_layout()
    plt.show()


def visualize_attention(attn_weights, src_tokens=None, tgt_tokens=None):
    """可视化attention权重矩阵。"""

    plt.figure(figsize=(10, 10))
    if torch.is_tensor(attn_weights):
        attn_weights = attn_weights.detach().cpu().numpy()
    sns.heatmap(
        attn_weights,
        cmap="YlOrRd",
        xticklabels=src_tokens if src_tokens else "auto",
        yticklabels=tgt_tokens if tgt_tokens else "auto",
        cbar_kws={"label": "Attention Weight"},
    )
    plt.xlabel("Source Sequence Position")
    plt.ylabel("Target Sequence Position")
    plt.title("Attention Weight Visualization\nDarker color indicates stronger attention")
    plt.tight_layout()
    plt.show()


# ===== 训练相关 =====
class Batch:
    """训练时的batch对象，构造mask与标签。"""

    def __init__(self, src, tgt=None, pad: int = 2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def rate(step, model_size, factor, warmup):
    """学习率调度函数，包含warmup与衰减。"""

    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def visualize_learning_rate_schedule():
    """可视化不同配置的学习率曲线。"""

    plt.figure(figsize=(12, 6))
    opts = [
        [512, 1, 4000],
        [512, 1, 8000],
        [256, 1, 4000],
    ]
    labels = [
        "Standard Config (d_model=512, warmup=4000)",
        "Long Warmup (d_model=512, warmup=8000)",
        "Small Model (d_model=256, warmup=4000)",
    ]

    for opt, label in zip(opts, labels):
        steps = list(range(1, 20000))
        rates = [rate(step, *opt) for step in steps]
        plt.plot(steps, rates, label=label)

    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedules with Different Configurations\nGradual decay after warmup")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


class LabelSmoothing(nn.Module):
    """Label Smoothing正则化。"""

    def __init__(self, size, padding_idx, smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def visualize_label_smoothing():
    """可视化label smoothing的目标分布。"""

    crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    target = torch.LongTensor([2, 1, 0])
    crit(predict.log(), target)

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.bar(range(5), crit.true_dist[i].numpy())
        plt.title(f"Sample {i + 1}: True Label={target[i].item()}")
        plt.xlabel("Class")
        plt.ylabel("Target Probability")
        plt.ylim([0, 1])
        plt.axhline(y=crit.confidence, color="r", linestyle="--", label=f"Confidence={crit.confidence:.2f}")
        plt.legend()
    plt.suptitle("Label Smoothing Effect\nCorrect class gets high probability, others get small probability", fontsize=14)
    plt.tight_layout()
    plt.show()


def data_gen(V, batch_size, nbatches):
    """复制任务的数据生成器。"""

    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, pad=0)


class SimpleLossCompute:
    """将生成器与损失组合，返回标量loss。"""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return sloss.data * norm, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """贪婪解码，每步取最大概率token。"""

    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def run_copy_task_example():
    """运行复制任务的小模型示例，返回训练好的模型。"""

    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400),
    )

    batch_size = 80
    losses = []
    model.train()
    for _ in range(20):
        epoch_loss = 0
        for batch in data_gen(V, batch_size, 20):
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = SimpleLossCompute(model.generator, criterion)(out, batch.tgt_y, batch.ntokens)
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            epoch_loss += loss
        losses.append(epoch_loss / 20)

    model.eval()
    return model, losses


# ===== 分析与可视化 =====
def analyze_model_structure(model):
    """打印模型参数统计并返回字典。"""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    components = {
        "Encoder": model.encoder,
        "Decoder": model.decoder,
        "Source Embedding": model.src_embed,
        "Target Embedding": model.tgt_embed,
        "Generator": model.generator,
    }
    component_params = {name: sum(p.numel() for p in comp.parameters()) for name, comp in components.items()}

    print("=" * 60)
    print("Transformer Model Structure Analysis")
    print("=" * 60)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("\nParameter Distribution by Component:")
    for name, params in component_params.items():
        print(f"{name:25s}: {params:10,} ({params / total_params * 100:5.2f}%)")

    plt.figure(figsize=(10, 6))
    plt.pie(component_params.values(), labels=component_params.keys(), autopct="%1.1f%%", startangle=90)
    plt.title("Parameter Distribution by Model Component")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "component_params": component_params,
    }


def visualize_model_attention(model, src, src_mask, tgt, tgt_mask):
    """可视化Encoder第一层各头的Self-Attention。"""

    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        model.decode(memory, src_mask, tgt, tgt_mask)

    if hasattr(model.encoder.layers[0].self_attn, "attn"):
        encoder_attn = model.encoder.layers[0].self_attn.attn
        if encoder_attn is None:
            print("Note: Attention weights not cached, need forward pass first.")
            return
        attn_weights = encoder_attn[0, 0].cpu().numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, cmap="YlOrRd", cbar_kws={"label": "Attention Weight"})
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.title("Encoder Self-Attention (Layer 1, Head 1)")
        plt.tight_layout()
        plt.show()

        n_heads = encoder_attn.shape[1]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i in range(min(8, n_heads)):
            attn_head = encoder_attn[0, i].cpu().numpy()
            sns.heatmap(attn_head, ax=axes[i], cmap="YlOrRd", cbar=False, square=True)
            axes[i].set_title(f"Head {i + 1}")
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
        plt.suptitle("Encoder Self-Attention - All Heads", fontsize=14)
        plt.tight_layout()
        plt.show()


def visualize_layer_outputs(model, src, src_mask):
    """逐层可视化Encoder输出特征图。"""

    model.eval()
    x = model.src_embed(src)
    layer_outputs = [x[0].detach().cpu().numpy()]
    for layer in model.encoder.layers:
        x = layer(x, src_mask)
        layer_outputs.append(x[0].detach().cpu().numpy())

    n_layers = len(layer_outputs)
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(16, 6))
    axes = axes.flatten()
    for i, output in enumerate(layer_outputs):
        im = axes[i].imshow(output.T, aspect="auto", cmap="RdBu", interpolation="nearest")
        axes[i].set_title(f"Layer {i} Output" if i > 0 else "Embedding")
        axes[i].set_xlabel("Sequence Position")
        axes[i].set_ylabel("Feature Dimension")
        plt.colorbar(im, ax=axes[i])
    for i in range(len(layer_outputs), len(axes)):
        axes[i].axis("off")
    plt.suptitle("Visualization of Encoder Layer Outputs", fontsize=14)
    plt.tight_layout()
    plt.show()


# ===== 模型保存/加载 =====
def save_model(model, path: str = "transformer_model.pt"):
    torch.save({"model_state_dict": model.state_dict()}, path)
    print(f"Model saved to: {path}")


def load_model(model, path: str = "transformer_model.pt"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {path}")
    return model


__all__ = [
    "device",
    "EncoderDecoder",
    "Generator",
    "clones",
    "LayerNorm",
    "SublayerConnection",
    "Encoder",
    "EncoderLayer",
    "Decoder",
    "DecoderLayer",
    "subsequent_mask",
    "attention",
    "MultiHeadedAttention",
    "PositionwiseFeedForward",
    "Embeddings",
    "PositionalEncoding",
    "make_model",
    "visualize_mask",
    "visualize_positional_encoding",
    "visualize_attention",
    "Batch",
    "rate",
    "visualize_learning_rate_schedule",
    "LabelSmoothing",
    "visualize_label_smoothing",
    "data_gen",
    "SimpleLossCompute",
    "greedy_decode",
    "run_copy_task_example",
    "analyze_model_structure",
    "visualize_model_attention",
    "visualize_layer_outputs",
    "save_model",
    "load_model",
]
