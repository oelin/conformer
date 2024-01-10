from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce, rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class RoPE(nn.Module):
    """Rotary positional embedding (RoPE).

    Rotary positional embedding (Su et al., 2023) rotates keys and queries by
    their absolute position such that their dot product depends only on their
    content and *relative position*. Generalized to arbitrary dimensions, RoPE
    divides a D-dimensional space into D//2 subspaces.

    Example
    -------
    >>> module = RoPE(embedding_dimension=256, base=10_000)
    >>> q = torch.randn((1, 10, 256))
    >>> k = torch.randn((1, 10, 256))
    >>> alignment = torch.einsum('bte,bse->bts', module(q), module(k))
    """

    def __init__(self, *, embedding_dimension: int, base: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        base : int
            The base to use for absolute positional encodings.
        """

        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.base = base

        # Precompute theta.

        exponent = torch.arange(
            start=0,
            end=embedding_dimension,
            step=2,
            dtype=torch.float,
        ) / embedding_dimension

        theta = 1. / torch.pow(base, exponent)

        self.theta = theta

    def absolute_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Perform absolute positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        encoding : torch.Tensor
            The absolute positional encoding.
        """

        if self.theta.device != x.device:
            self.theta = self.theta.to(x.device)

        encoding = torch.einsum(
            't,e->te',
            torch.arange(x.size(-2), dtype=torch.float, device=x.device),
            self.theta,
        )

        encoding = repeat(encoding, '... e -> ... (e n)', n=2)

        return encoding

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate each subspace by -90 degrees."""

        x = rearrange(x, '... (e n) -> ... e n', n=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = rearrange(x, '... e n -> ... (e n)')

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Foward pass."""

        encoding = self.absolute_positional_encoding(x)
        x = x * encoding.cos() + (self.rotate_half(x) * encoding.sin())

        return x


class Attention(nn.Module):
    """Attention.

    Implements multi-head self attention (Vaswani et al., 2017) with rotary
    positional embedding (RoPE) (Lu et al., 2021).

    Example
    -------
    >>> module = Attention(
    ...    embedding_dimension=256,
    ...    number_of_heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        number_of_heads: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of heads.
        """

        super().__init__()

        self.number_of_heads = number_of_heads

        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.rope = RoPE(
            embedding_dimension=embedding_dimension // number_of_heads,
            base=10_000,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        h = self.number_of_heads
        q, k, v = rearrange(self.linear_1(x), 'b s (n h e) -> n b h s e', n=3, h=h)
        q, k = self.rope(q), self.rope(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = self.linear_2(rearrange(x, 'b h s e -> b s (h e)'))

        return x


class AttentionBlock(nn.Module):
    """Attention block.

    Implements the Conformer attention block.

    Example
    -------
    >>> module = AttentionBlock(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ...     dropout_probability=0.,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        number_of_heads: int,
        dropout_probability: float,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of heads.
        dropout_probability : float
            The dropout probability.
        """

        super().__init__()

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dimension,
        )

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )

        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : Optional[torch.Tensor]
            The attention mask.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        z = self.layer_norm(x)
        z = self.attention(z, mask=mask)
        z = self.dropout(z)

        return x + z


class ConvolutionBlock(nn.Module):
    """Convolution block.

    Implements the Conformer convolution module.

    Example
    -------
    >>> module = ConvolutionBlock(
    ...     embedding_dimension=256,
    ...     dropout_probability=0.,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        dropout_probability: float,
    ) -> None:
        """Initialize the module

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        dropout_probability : float
            The dropout probability.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dimension),
            Rearrange('b t e -> b e t'),
            nn.Conv1d(
                in_channels=embedding_dimension,
                out_channels=embedding_dimension * 2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=embedding_dimension,
                out_channels=embedding_dimension,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=embedding_dimension,
                bias=False,
            ),
            nn.SiLU(),
            nn.GroupNorm(
                num_groups=32,
                num_channels=embedding_dimension,
            ),
            nn.Conv1d(
                in_channels=embedding_dimension,
                out_channels=embedding_dimension,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Rearrange('b e t -> b t e'),
            nn.Dropout(p=dropout_probability),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        return x + self.layers(x)


class FeedForwardBlock(nn.Module):
    """Feed-forward block.

    Implements the conformer feed-forward block.

    Example
    -------
    >>> module = FeedForwardBlock(
    ...     embedding_dimension=256,
    ...     dropout_probability=0.,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        dropout_probability: float,
    ) -> None:
        """Initialize the module

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        dropout_probability : float
            The dropout probability.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=embedding_dimension,
            ),
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * 4,
            ),
            nn.SiLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(
                in_features=embedding_dimension * 4,
                out_features=embedding_dimension,
            ),
            nn.Dropout(p=dropout_probability),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        return x + self.layers(x)


class ConformerBlock(nn.Module):
    """Conformer block.

    Implements the Conformer block.

    Example
    -------
    >>> module = ConformerBlock(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ...     dropout_probability=0.,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        dropout_probability: int,
    ) -> None:
        """Initialize the module

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of heads.
        dropout_probability : float
            The dropout probability.
        """

        super().__init__()

        self.blocks = nn.ModuleList([
            FeedForwardBlock(
                embedding_dimension=embedding_dimension,
                dropout_probability=dropout_probability,
            ),
            AttentionBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                dropout_probability=dropout_probability,
            ),
            ConvolutionBlock(
                embedding_dimension=embedding_dimension,
                dropout_probability=dropout_probability,
            ),
            FeedForwardBlock(
                embedding_dimension=embedding_dimension,
                dropout_probability=dropout_probability,
            ),
        ])

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dimension,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : Optional[torch.Tensor]
            The attention mask.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = x + self.blocks[0](x)
        x = x + self.blocks[1](x, mask=mask)
        x = x + self.blocks[2](x)
        x = x + self.blocks[3](x)

        x = self.layer_norm(x)

        return x


@dataclass(frozen=True)
class ConformerConfiguration:
    embedding_dimension: int
    number_of_heads: int
    number_of_layers: int
    dropout_probability: int


class Conformer(nn.Module):
    """Conformer.

    Implements Conformer (Gulati at al., 2020). Conformer is a 
    convolution-augmented transformer architecture intended for ASR tasks. Note
    that this implementation does not include the preprocessing stack.

    Example
    -------
    >>> configuration = ConformerConfiguration(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ...     number_of_layers=8,
    ...     dropout_probability=0.,
    ... )
    >>> model = Conformer(configuration=configuration)
    >>> x = torch.randn((1, 10, 256))
    >>> x = model(x)
    """

    def __init__(self, configuration: ConformerConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : ConformerConfiguration
            The configuration.
        """

        super().__init__()

        self.layers = nn.Sequential(
            ConformerBlock(
                embedding_dimension=configuration.embedding_dimension,
                number_of_heads=configuration.number_of_heads,
                dropout_probability=configuration.dropout_probability,
            ) for _ in range(configuration.number_of_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        return self.layers(x)
