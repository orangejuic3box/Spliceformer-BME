"""
model.py

Neural architectures for splice-site prediction:

  • Low-level building blocks:
        - FeedForward: MLP block used inside Transformer layers.
        - Attention: multi-head self-attention with a gating mechanism.
        - FixedPositionalEmbedding: sinusoidal positional embeddings.
        - ResidualBlock / ResComboBlock: 1D convolutional residual units.

  • Policy:
        Small head that outputs per-position logits for a 2-action policy
        (used to select acceptor/donor positions).

  • Transformer:
        Sequence Transformer operating on a subset of positions (selected
        via actions), then expanded back to full length.

  • SpliceAI / SpliceAI_10K / SpliceAI_small:
        CNN-style architectures that resemble SpliceAI with dilated residual
        blocks and skip connections.

  • SpliceFormer:
        Main model: SpliceAI-style front-end, followed by a policy network
        and several Transformer blocks that refine selected positions. Produces
        3-class splice-site probabilities per position.

  • ResNet_40K:
        Deeper residual model with optional exon-inclusion head.
"""

import torch
import numpy as np
from torch import nn
from einops import rearrange
from .weight_init import keras_init


def pair(t):
    """
    Utility to convert a scalar into a 2-tuple.

    Example:
      pair(3) → (3, 3)
      pair((2, 5)) → (2, 5)
    """
    return t if isinstance(t, tuple) else (t, t)


# ---------------------------------------------------------------------------
# Basic Transformer blocks
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Simple position-wise feed-forward MLP used inside Transformer blocks.

    Architecture:
      Linear(dim → hidden_dim) → GELU → Dropout
      → Linear(hidden_dim → dim) → Dropout
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, dim)
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention with a learnable gating mechanism.

    Inputs
    ------
    x : (B, N, dim)
        Sequence of N tokens, each of dimension `dim`.

    Outputs
    -------
    out : (B, N, dim)
        Attention-aggregated representation, possibly with linear projection.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        # If we have a single head with dim_head == dim, we can skip
        # the final projection.
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # Single linear to generate Q, K, V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Gate to modulate the attention output with a function of x
        self.gate = nn.Linear(dim, inner_dim)

        # Optional final projection back to dim
        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        # Compute Q, K, V and reshape to (B, heads, N, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv,
        )

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        # Weighted sum of values
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Apply gating based on the original input
        out = torch.sigmoid(self.gate(x)) * out
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Policy network for selecting positions (acceptor / donor)
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    """
    Small per-position policy network.

    For each position, takes an n_channels-dimensional feature vector and
    outputs 2 logits: (acceptor_score, donor_score).

    Used in SpliceFormer to select which positions a Transformer should focus on.
    """

    def __init__(self, n_channels):
        super(Policy, self).__init__()
        self.n_channels = n_channels
        self.affine1 = nn.Linear(n_channels, 4)
        self.act = nn.LeakyReLU()
        self.affine2 = nn.Linear(4, 2)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, C, L)
            Feature maps (channels-first).

        Returns
        -------
        action_scores : torch.Tensor, shape (B, L, 2)
            Unnormalized logits for two actions per position.
        """
        # Switch to (B, L, C) for linear layer over channels
        x = torch.transpose(x, 1, 2)
        x = self.affine1(x) / np.sqrt(self.n_channels)
        x = self.act(x)
        action_scores = self.affine2(x)
        return action_scores


# ---------------------------------------------------------------------------
# Transformer that acts on a subset of positions
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Transformer module that operates only on a subset of positions
    specified by `actions`, then expands back to the full sequence.

    Steps:
      1. Add positional embeddings to the full sequence.
      2. Gather a subset of positions using `actions`.
      3. Apply multiple layers of (LayerNorm → Attention / FF + residual).
      4. Expand the updated subset back to the full length using scatter.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.layerNormLayers = nn.ModuleList([])
        self.pos_emb = FixedPositionalEmbedding(dim)

        for _ in range(depth):
            # Each layer has two LayerNorms: one for attention, one for FF
            self.layerNormLayers.append(
                nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])
            )
            # Attention + feed-forward
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, state, actions):
        """
        Parameters
        ----------
        state : torch.Tensor, shape (B, C, L)
            Input feature map.
        actions : torch.Tensor, shape (B, K, C)
            Indices (per batch) into positions to gather, repeated to
            have the same channel dimension for gather().

        Returns
        -------
        torch.Tensor, shape (B, C, L)
            Feature map after Transformer refinement.
        """
        # Convert to (B, L, C) for transformer
        x_in = torch.transpose(state, 1, 2)

        # Positional embeddings: (1, L, C)
        pos_emb = self.pos_emb(x_in).type_as(x_in)

        # Gather subset of positions to process: (B, K, C)
        x_in_subset = torch.gather(x_in + pos_emb, 1, actions)
        x = x_in_subset

        # Apply each Transformer layer (pre-norm + residual)
        for d, (attn, ff) in enumerate(self.layers):
            x = attn(self.layerNormLayers[d][0](x)) + x
            x = ff(self.layerNormLayers[d][1](x)) + x

        # Expand subset back into the original full tensor using scatter
        x = self.expand_sub_tensor(x - x_in_subset, x_in, actions)
        return torch.transpose(x, 1, 2)

    def top_k_selection(self, x, conv_final, m_1, k):
        """
        Select top-k positions based on acceptor/donor scores.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, L)
        conv_final : nn.Conv1d
            Final conv layer that maps to 3-class logits.
        m_1 : nn.Softmax
            Softmax over class dimension.
        k : int
            Total number of positions to select (half acceptors, half donors).

        Returns
        -------
        x_subset : torch.Tensor, shape (B, k, C)
        splice_idx : torch.Tensor, shape (B, k, C)
            Indices (expanded along channels) of selected positions.
        """
        attention = m_1(conv_final(torch.transpose(x, 1, 2)))
        acceptors = attention[:, 1, :]
        donors = attention[:, 2, :]

        # Top-k/2 acceptor positions
        acceptors_sorted = torch.argsort(
            acceptors, dim=1, descending=True
        )[:, : k // 2]
        # Top-k/2 donor positions
        donors_sorted = torch.argsort(
            donors, dim=1, descending=True
        )[:, : k // 2]

        # Combine and sort indices
        splice_idx = torch.cat([acceptors_sorted, donors_sorted], dim=1)
        splice_idx = torch.sort(splice_idx, dim=1).values
        splice_idx = splice_idx.unsqueeze(2).repeat(1, 1, self.dim)

        return torch.gather(x, 1, splice_idx), splice_idx

    def expand_sub_tensor(self, x_subset, x, splice_idx):
        """
        Scatter the subset tensor back into the full-length tensor.

        Parameters
        ----------
        x_subset : torch.Tensor, shape (B, K, C)
            Updated features at selected positions.
        x : torch.Tensor, shape (B, L, C)
            Original full-length features.
        splice_idx : torch.Tensor, shape (B, K, C)
            Indices for scatter.

        Returns
        -------
        torch.Tensor, shape (B, L, C)
            x + scattered x_subset.
        """
        tmp = torch.zeros_like(x)
        return tmp.scatter_(1, splice_idx, x_subset) + x


class FixedPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings as used in Transformers.

    For each position i and channel dimension j:

      PE[i, 2j]   = sin(i / 10000^(2j / dim))
      PE[i, 2j+1] = cos(i / 10000^(2j / dim))
    """

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (B, L, C) → positions dimension is L
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        # (1, L, C)
        return emb[None, :, :]


# ---------------------------------------------------------------------------
# Conv / residual building blocks (SpliceAI-like)
# ---------------------------------------------------------------------------

def activation_func(activation):
    """
    Map activation name to a module instance.
    """
    return nn.ModuleDict(
        [
            ["relu", nn.ReLU(inplace=True)],
            ["leaky_relu", nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ["selu", nn.SELU(inplace=True)],
            ["none", nn.Identity()],
        ]
    )[activation]


class ResidualBlock(nn.Module):
    """
    1D residual block with two Conv1d + BatchNorm + activation layers.

    The input and output channel counts are equal, so the shortcut is identity.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        activation="relu",
        bn_momentum=0.01,
    ):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = (
            in_channels,
            out_channels,
            activation,
        )
        paddingAmount = int(dilation * (kernel_size - 1) / 2)

        self.convlayer1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            padding=paddingAmount,
            padding_mode="zeros",
        )
        self.convlayer2 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            padding=paddingAmount,
            padding_mode="zeros",
        )
        self.activate = activation_func(activation)
        self.bn1 = nn.BatchNorm1d(self.in_channels, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm1d(self.in_channels, momentum=bn_momentum)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.activate(self.bn1(x))
        x = self.convlayer1(x)
        x = self.activate(self.bn2(x))
        x = self.convlayer2(x)
        x += residual
        return x


class ResComboBlock(nn.Module):
    """
    Combination of four identical ResidualBlocks in sequence.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res_W,
        res_dilation,
        bn_momentum=0.01,
    ):
        super().__init__()
        self.comboBlock = nn.Sequential(
            ResidualBlock(
                in_channels,
                out_channels,
                res_W,
                res_dilation,
                bn_momentum=bn_momentum,
            ),
            ResidualBlock(
                in_channels,
                out_channels,
                res_W,
                res_dilation,
                bn_momentum=bn_momentum,
            ),
            ResidualBlock(
                in_channels,
                out_channels,
                res_W,
                res_dilation,
                bn_momentum=bn_momentum,
            ),
            ResidualBlock(
                in_channels,
                out_channels,
                res_W,
                res_dilation,
                bn_momentum=bn_momentum,
            ),
        )

    def forward(self, x):
        return self.comboBlock(x)


class SpliceAI(nn.Module):
    """
    SpliceAI-like CNN front-end.

    Takes a 4-channel one-hot sequence and produces a 32-channel feature map
    with long-range context via dilated residual blocks and skip connections.

    The output is not cropped to remove CL_max/2 on each side; that is left
    to downstream modules.
    """

    def __init__(self, CL_max, bn_momentum=0.01, **kwargs):
        super().__init__()
        self.n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11, 11, 21, 41]
        res_dilation = [1, 4, 10, 25]
        self.kernel_size = 1

        # Initial conv: 4 input channels (A,C,G,T) → n_channels
        self.conv_layer_1 = nn.Conv1d(
            in_channels=4,
            out_channels=self.n_channels,
            kernel_size=self.kernel_size,
            stride=1,
        )

        # Skip connections at multiple depths
        self.skip_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.n_channels,
                    out_channels=self.n_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
                for i in range(5)
            ]
        )

        # Four ResComboBlocks with increasing dilation
        self.res_layers = nn.ModuleList(
            [
                ResComboBlock(
                    in_channels=self.n_channels,
                    out_channels=self.n_channels,
                    res_W=self.res_W[i],
                    res_dilation=res_dilation[i],
                    bn_momentum=bn_momentum,
                )
                for i in range(4)
            ]
        )

    def forward(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor, shape (B, 4, L)
            One-hot sequence.

        Returns
        -------
        x_skip : torch.Tensor, shape (B, 32, L)
            Feature map after skip aggregation (no cropping).
        """
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i, residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            skip += self.skip_layers[i + 1](x)

        x_skip = skip[:, :, :]
        return x_skip


# ---------------------------------------------------------------------------
# SpliceFormer: SpliceAI + policy + Transformer refinement
# ---------------------------------------------------------------------------

class SpliceFormer(nn.Module):
    """
    SpliceFormer model.

    Pipeline:
      1) SpliceAI front-end to compute deep sequence features.
      2) Policy network to select acceptor/donor positions.
      3) Several Transformer blocks that act only on these positions
         and scatter their updates back into the full sequence.
      4) Final Conv1d → Softmax over 3 output classes at each position.

    Optionally returns policy actions and feature maps for RL-style training.
    """

    def __init__(
        self,
        CL_max,
        n_channels=32,
        maxSeqLength=4 * 128,
        depth=4,
        n_transformer_blocks=2,
        heads=4,
        dim_head=32,
        mlp_dim=512,
        dropout=0.01,
        returnFmap=False,
        bn_momentum=0.01,
        determenistic=False,
        crop=True,
        **kwargs,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.CL_max = CL_max
        self.kernel_size = 1
        self.crop = crop
        self.returnFmap = returnFmap

        # SpliceAI front-end, with Keras-like initialization
        self.SpliceAI = SpliceAI(CL_max, bn_momentum=bn_momentum).apply(
            keras_init
        )

        # Final conv over channels to get 3-class logits per position
        self.conv_final = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=3,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.softmax = nn.Softmax(dim=1)

        self.maxSeqLength = maxSeqLength
        self.policy = Policy(n_channels=n_channels)
        self.determenistic = determenistic

        # Skip connections per Transformer block
        self.skip_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.n_channels,
                    out_channels=self.n_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
                for i in range(n_transformer_blocks + 1)
            ]
        )

        # List of Transformer blocks acting on subsets of positions
        self.transformerBlocks = nn.ModuleList(
            [
                Transformer(
                    self.n_channels,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for i in range(n_transformer_blocks)
            ]
        )

    def forward(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor, shape (B, 4, L)
            One-hot input sequence.

        Returns
        -------
        out : torch.Tensor, shape (B, 3, L_trimmed or L)
            3-class probabilities per position.
        acceptor_actions, donor_actions : torch.Tensor
            Selected positions for acceptors/donors.
        acceptor_log_probs, donor_log_probs : torch.Tensor
            Log-probabilities under the policy for those actions.
        (optionally) fmaps : torch.Tensor
            Cropped feature maps if returnFmap=True.
        """
        # SpliceAI front-end features
        state = self.SpliceAI(features)
        m_1 = self.softmax

        # Policy-based selection of positions to refine
        (
            actions,
            acceptor_actions,
            donor_actions,
            acceptor_log_probs,
            donor_log_probs,
        ) = self.select_action(state)

        x = state
        skip = self.skip_layers[0](x)

        # Apply sequence of Transformer blocks and accumulate skip features
        for i, transformer in enumerate(self.transformerBlocks):
            x = transformer(x, actions)
            skip += self.skip_layers[i + 1](x)

        # Final classification layer
        out = m_1(self.conv_final(skip))

        # Optional cropping to remove CL_max/2 context on each side
        if self.crop:
            out = out[:, :, (self.CL_max // 2) : -(self.CL_max // 2)]

        if self.returnFmap:
            fmaps = skip[:, :, (self.CL_max // 2) : -(self.CL_max // 2)]
            return (
                out,
                acceptor_actions,
                donor_actions,
                acceptor_log_probs,
                donor_log_probs,
                fmaps,
            )
        else:
            return (
                out,
                acceptor_actions,
                donor_actions,
                acceptor_log_probs,
                donor_log_probs,
            )

    def select_action(self, state):
        """
        Select positions for acceptor/donor refinement.

        Two modes:
          • determenistic=True:
                - Rank positions by policy logits for acceptor/donor separately.
                - Enforce that each position is assigned to at most one type.
                - Take top maxSeqLength/2 positions for each.

          • determenistic=False:
                - Sample positions from categorical distributions over
                  policy_logits[:,:,0] (acceptor) and policy_logits[:,:,1]
                  (donor), masking already chosen positions.

        Returns
        -------
        actions : torch.Tensor, shape (B, K, C)
            Indices (per batch) expanded along channels for gathering.
        acceptor_actions, donor_actions : torch.Tensor
            Chosen positions for each class.
        acceptor_log_probs, donor_log_probs : torch.Tensor
            Log-probabilities for the chosen actions.
        """
        # c tracks which positions have been selected (mask)
        c = state[:, 0, :].new_zeros(state.size(0), state.size(2))
        policy_logits = self.policy(state)

        actions = []
        acceptor_actions = []
        donor_actions = []
        acceptor_log_probs = []
        donor_log_probs = []

        if self.determenistic:
            # Deterministic: pick top positions without replacement
            acceptor_logits = policy_logits[:, :, 0]
            donor_logits = policy_logits[:, :, 1]

            # Double ranking trick to enforce disjoint positions
            acceptor_order = torch.argsort(
                torch.argsort(acceptor_logits, dim=1), dim=1
            )
            donor_order = torch.argsort(
                torch.argsort(donor_logits, dim=1), dim=1
            )

            # Enforce that acceptor/donor sets do not overlap
            acceptor_logits[acceptor_order - donor_order < 0] = -float("inf")
            donor_logits[donor_order - acceptor_order <= 0] = -float("inf")

            acceptor_log_probs, acceptor_actions = torch.topk(
                acceptor_logits,
                self.maxSeqLength // 2,
                dim=1,
                largest=True,
                sorted=False,
            )
            donor_log_probs, donor_actions = torch.topk(
                donor_logits,
                self.maxSeqLength // 2,
                dim=1,
                largest=True,
                sorted=False,
            )

            actions = torch.cat([acceptor_actions, donor_actions], dim=1)
            # Expand to have channel dimension for gather
            return (
                actions.unsqueeze(2).repeat(1, 1, self.n_channels),
                acceptor_actions,
                donor_actions,
                acceptor_log_probs,
                donor_log_probs,
            )
        else:
            # Stochastic: sample positions sequentially, masking used ones
            for i in range(self.maxSeqLength // 2):
                # Acceptors
                m = torch.distributions.Categorical(
                    logits=policy_logits[:, :, 0]
                    - torch.nan_to_num(float("inf") * c, nan=0)
                )
                action = m.sample()
                acceptor_log_probs.append(m.log_prob(action).unsqueeze(1))
                actions.append(action.unsqueeze(1))
                acceptor_actions.append(action.unsqueeze(1))
                c[torch.arange(c.size(0), dtype=torch.long), action] = 1

                # Donors
                m = torch.distributions.Categorical(
                    logits=policy_logits[:, :, 1]
                    - torch.nan_to_num(float("inf") * c, nan=0)
                )
                action = m.sample()
                donor_log_probs.append(m.log_prob(action).unsqueeze(1))
                actions.append(action.unsqueeze(1))
                donor_actions.append(action.unsqueeze(1))
                c[torch.arange(c.size(0), dtype=torch.long), action] = 1

            return (
                torch.hstack(actions).unsqueeze(2).repeat(1, 1, self.n_channels),
                torch.hstack(acceptor_actions),
                torch.hstack(donor_actions),
                torch.hstack(acceptor_log_probs),
                torch.hstack(donor_log_probs),
            )


# ---------------------------------------------------------------------------
# Additional architectures: SpliceAI_10K, SpliceAI_small, ResNet_40K
# ---------------------------------------------------------------------------

class SpliceAI_10K(nn.Module):
    """
    Standalone SpliceAI-style model with explicit 3-class output and
    10k context length (CL_max).

    Outputs a 3-class softmax over positions after cropping context.
    """

    def __init__(self, CL_max, **kwargs):
        super().__init__()
        n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11, 11, 21, 41]
        res_dilation = [1, 4, 10, 25]
        self.kernel_size = 1

        self.conv_layer_1 = nn.Conv1d(
            in_channels=4,
            out_channels=n_channels,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.skip_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
                for i in range(5)
            ]
        )
        self.res_layers = nn.ModuleList(
            [
                ResComboBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    res_W=self.res_W[i],
                    res_dilation=res_dilation[i],
                )
                for i in range(4)
            ]
        )
        self.conv_final = nn.Conv1d(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i, residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            skip += self.skip_layers[i + 1](x)

        # Crop context (CL_max/2 on each side)
        x = skip[:, :, self.CL_max // 2 : -self.CL_max // 2]
        x = self.conv_final(x)
        return self.softmax(x)


class SpliceAI_small(nn.Module):
    """
    Smaller SpliceAI-style model with only one ResComboBlock.

    Intended as a lighter-weight feature extractor.
    """

    def __init__(self, CL_max, **kwargs):
        super().__init__()
        n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11]
        res_dilation = [1]
        self.kernel_size = 1

        self.conv_layer_1 = nn.Conv1d(
            in_channels=4,
            out_channels=n_channels,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.skip_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
                for i in range(2)
            ]
        )
        self.res_layers = nn.ModuleList(
            [
                ResComboBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    res_W=self.res_W[i],
                    res_dilation=res_dilation[i],
                )
                for i in range(1)
            ]
        )
        self.conv_final = nn.Conv1d(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=self.kernel_size,
            stride=1,
        )

    def forward(self, features):
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i, residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            skip += self.skip_layers[i + 1](x)

        # No softmax/cropping here; returns features with residual skips
        return skip


class ResNet_40K(nn.Module):
    """
    Deeper residual network with ~40k context and optional exon-inclusion head.

    Outputs:
      • 3-class splice-site probabilities over positions (softmax).
      • Optionally, a 1-channel exon-inclusion probability (sigmoid).
    """

    def __init__(self, CL_max, exonInclusion=False, **kwargs):
        super().__init__()
        n_channels = 32
        self.CL_max = CL_max
        self.res_W = [11, 11, 21, 41, 51]
        res_dilation = [1, 4, 10, 25, 75]
        self.kernel_size = 1
        self.exonInclusion = exonInclusion

        self.conv_layer_1 = nn.Conv1d(
            in_channels=4,
            out_channels=n_channels,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.skip_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
                for i in range(6)
            ]
        )
        self.res_layers = nn.ModuleList(
            [
                ResComboBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    res_W=self.res_W[i],
                    res_dilation=res_dilation[i],
                )
                for i in range(5)
            ]
        )
        self.conv_final = nn.Conv1d(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.softmax = nn.Softmax(dim=1)

        if exonInclusion:
            self.conv_exon = nn.Conv1d(
                in_channels=n_channels,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=1,
            )

    def forward(self, features):
        x = self.conv_layer_1(features)
        skip = self.skip_layers[0](x)

        for i, residualUnit in enumerate(self.res_layers):
            x = residualUnit(x)
            skip += self.skip_layers[i + 1](x)

        # Crop context
        x = skip[:, :, self.CL_max // 2 : -self.CL_max // 2]
        out = self.softmax(self.conv_final(x))

        if self.exonInclusion:
            exon = nn.Sigmoid()(self.conv_exon(x))
            return out, exon
        else:
            return out
