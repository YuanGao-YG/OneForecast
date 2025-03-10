from typing import Any, List, Optional

import dgl
import torch
from dgl import DGLGraph
from torch import Tensor

try:
    from typing import Self
except ImportError:
    # for Python versions < 3.11
    from typing_extensions import Self


try:
    from pylibcugraphops.pytorch import BipartiteCSC, StaticCSC

    USE_CUGRAPHOPS = True

except ImportError:
    StaticCSC = None
    BipartiteCSC = None
    USE_CUGRAPHOPS = False


class CuGraphCSC:

    def __init__(
        self,
        offsets: Tensor,
        indices: Tensor,
        num_src_nodes: int,
        num_dst_nodes: int,
        ef_indices: Optional[Tensor] = None,
        reverse_graph_bwd: bool = True,
        cache_graph: bool = True,
        partition_size: Optional[int] = -1,
        partition_group_name: Optional[str] = None,
    ) -> None:
        self.offsets = offsets
        self.indices = indices
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.ef_indices = ef_indices
        self.reverse_graph_bwd = reverse_graph_bwd
        self.cache_graph = cache_graph

        # cugraph-ops structures
        self.bipartite_csc = None
        self.static_csc = None
        # dgl graph
        self.dgl_graph = None

        self.is_distributed = False
        self.dist_csc = None

        if partition_size <= 1:
            self.is_distributed = False
            return

        if self.ef_indices is not None:
            raise AssertionError(
                "DistributedGraph does not support mapping CSC-indices to COO-indices."
            )


        # overwrite graph information with local graph after distribution
        self.offsets = self.dist_graph.graph_partition.local_offsets
        self.indices = self.dist_graph.graph_partition.local_indices
        self.num_src_nodes = self.dist_graph.graph_partition.num_local_src_nodes
        self.num_dst_nodes = self.dist_graph.graph_partition.num_local_dst_nodes
        self.is_distributed = True

    @staticmethod
    def from_dgl(
        graph: DGLGraph,
        partition_size: int = 1,
        partition_group_name: Optional[str] = None,
        partition_by_bbox: bool = False,
        src_coordinates: Optional[torch.Tensor] = None,
        dst_coordinates: Optional[torch.Tensor] = None,
        coordinate_separators_min: Optional[List[List[Optional[float]]]] = None,
        coordinate_separators_max: Optional[List[List[Optional[float]]]] = None,
    ):  # pragma: no cover
        # DGL changed their APIs w.r.t. how sparse formats can be accessed
        # this here is done to support both versions
        if hasattr(graph, "adj_tensors"):
            offsets, indices, edge_perm = graph.adj_tensors("csc")
        elif hasattr(graph, "adj_sparse"):
            offsets, indices, edge_perm = graph.adj_sparse("csc")
        else:
            raise ValueError("Passed graph object doesn't support conversion to CSC.")

        n_src_nodes, n_dst_nodes = (graph.num_src_nodes(), graph.num_dst_nodes())

        graph_partition = None


        graph_csc = CuGraphCSC(
            offsets.to(dtype=torch.int64),
            indices.to(dtype=torch.int64),
            n_src_nodes,
            n_dst_nodes,
            partition_size=partition_size,
            partition_group_name=partition_group_name,
            graph_partition=graph_partition,
        )

        return graph_csc, edge_perm


    def to(self, *args: Any, **kwargs: Any) -> Self:
       
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype not in (
            None,
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                f"Invalid dtype, expected torch.int32 or torch.int64, got {dtype}."
            )
        self.offsets = self.offsets.to(device=device, dtype=dtype)
        self.indices = self.indices.to(device=device, dtype=dtype)
        if self.ef_indices is not None:
            self.ef_indices = self.ef_indices.to(device=device, dtype=dtype)

        return self

    def to_bipartite_csc(self, dtype=None) -> BipartiteCSC:
        
        if not (USE_CUGRAPHOPS):
            raise RuntimeError(
                "Conversion failed, expected cugraph-ops to be installed."
            )
        if not self.offsets.is_cuda:
            raise RuntimeError("Expected the graph structures to reside on GPU.")

        if self.bipartite_csc is None or not self.cache_graph:
           
            graph_offsets = self.offsets
            graph_indices = self.indices
            graph_ef_indices = self.ef_indices

            if dtype is not None:
                graph_offsets = self.offsets.to(dtype=dtype)
                graph_indices = self.indices.to(dtype=dtype)
                if self.ef_indices is not None:
                    graph_ef_indices = self.ef_indices.to(dtype=dtype)

            graph = BipartiteCSC(
                graph_offsets,
                graph_indices,
                self.num_src_nodes,
                graph_ef_indices,
                reverse_graph_bwd=self.reverse_graph_bwd,
            )
            self.bipartite_csc = graph

        return self.bipartite_csc

    def to_static_csc(self, dtype=None) -> StaticCSC:
       

        if not (USE_CUGRAPHOPS):
            raise RuntimeError(
                "Conversion failed, expected cugraph-ops to be installed."
            )
        if not self.offsets.is_cuda:
            raise RuntimeError("Expected the graph structures to reside on GPU.")

        if self.static_csc is None or not self.cache_graph:
           
            graph_offsets = self.offsets
            graph_indices = self.indices
            graph_ef_indices = self.ef_indices

            if dtype is not None:
                graph_offsets = self.offsets.to(dtype=dtype)
                graph_indices = self.indices.to(dtype=dtype)
                if self.ef_indices is not None:
                    graph_ef_indices = self.ef_indices.to(dtype=dtype)

            graph = StaticCSC(
                graph_offsets,
                graph_indices,
                graph_ef_indices,
            )
            self.static_csc = graph

        return self.static_csc

    def to_dgl_graph(self) -> DGLGraph:  # pragma: no cover

        if self.dgl_graph is None or not self.cache_graph:
            if self.ef_indices is not None:
                raise AssertionError("ef_indices is not supported.")
            graph_offsets = self.offsets
            dst_degree = graph_offsets[1:] - graph_offsets[:-1]
            src_indices = self.indices
            dst_indices = torch.arange(
                0,
                graph_offsets.size(0) - 1,
                dtype=graph_offsets.dtype,
                device=graph_offsets.device,
            )
            dst_indices = torch.repeat_interleave(dst_indices, dst_degree, dim=0)

            # labels not important here
            self.dgl_graph = dgl.heterograph(
                {("src", "src2dst", "dst"): ("coo", (src_indices, dst_indices))},
                idtype=torch.int32,
            )

        return self.dgl_graph


from typing import Any, Callable, Dict, Tuple, Union

import dgl.function as fn
import torch
from dgl import DGLGraph
from torch import Tensor
from torch.utils.checkpoint import checkpoint


try:
    from pylibcugraphops.pytorch.operators import (
        agg_concat_e2n,
        update_efeat_bipartite_e2e,
        update_efeat_static_e2e,
    )

    USE_CUGRAPHOPS = True

except ImportError:
    update_efeat_bipartite_e2e = None
    update_efeat_static_e2e = None
    agg_concat_e2n = None
    USE_CUGRAPHOPS = False


def checkpoint_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
   
    return layer(*args)


def set_checkpoint_fn(do_checkpointing: bool) -> Callable:
   
    if do_checkpointing:
        return checkpoint
    else:
        return checkpoint_identity


def concat_message_function(edges: Tensor) -> Dict[str, Tensor]:
   
    # concats src node , dst node, and edge features
    cat_feat = torch.cat((edges.data["x"], edges.src["x"], edges.dst["x"]), dim=1)
    return {"cat_feat": cat_feat}


@torch.jit.ignore()
def concat_efeat_dgl(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[torch.Tensor, torch.Tensor]],
    graph: DGLGraph,
) -> Tensor:
    
    if isinstance(nfeat, Tuple):
        src_feat, dst_feat = nfeat
        with graph.local_scope():
            graph.srcdata["x"] = src_feat
            graph.dstdata["x"] = dst_feat
            graph.edata["x"] = efeat
            graph.apply_edges(concat_message_function)
            return graph.edata["cat_feat"]

    with graph.local_scope():
        graph.ndata["x"] = nfeat
        graph.edata["x"] = efeat
        graph.apply_edges(concat_message_function)
        return graph.edata["cat_feat"]


def concat_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor]],
    graph: Union[DGLGraph, CuGraphCSC],
) -> Tensor:
   
    if isinstance(nfeat, Tensor):
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                src_feat, dst_feat = nfeat, nfeat
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                efeat = concat_efeat_dgl(
                    efeat, (src_feat, dst_feat), graph.to_dgl_graph()
                )

            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                    # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
                    bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
                    dst_feat = nfeat
                    efeat = update_efeat_bipartite_e2e(
                        efeat, src_feat, dst_feat, bipartite_graph, "concat"
                    )

                else:
                    static_graph = graph.to_static_csc()
                    efeat = update_efeat_static_e2e(
                        efeat,
                        nfeat,
                        static_graph,
                        mode="concat",
                        use_source_emb=True,
                        use_target_emb=True,
                    )

        else:
            efeat = concat_efeat_dgl(efeat, nfeat, graph)

    else:
        src_feat, dst_feat = nfeat
        # update edge features through concatenating edge and node features
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                efeat = concat_efeat_dgl(
                    efeat, (src_feat, dst_feat), graph.to_dgl_graph()
                )

            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
                bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
                efeat = update_efeat_bipartite_e2e(
                    efeat, src_feat, dst_feat, bipartite_graph, "concat"
                )
        else:
            efeat = concat_efeat_dgl(efeat, (src_feat, dst_feat), graph)

    return efeat


@torch.jit.script
def sum_efeat_dgl(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, src_idx: Tensor, dst_idx: Tensor
) -> Tensor:
    

    return efeat + src_feat[src_idx] + dst_feat[dst_idx]


def sum_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor]],
    graph: Union[DGLGraph, CuGraphCSC],
):
   
    if isinstance(nfeat, Tensor):
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                src_feat, dst_feat = nfeat, nfeat
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                src, dst = (item.long() for item in graph.to_dgl_graph().edges())
                sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                    dst_feat = nfeat
                    bipartite_graph = graph.to_bipartite_csc()
                    sum_efeat = update_efeat_bipartite_e2e(
                        efeat, src_feat, dst_feat, bipartite_graph, mode="sum"
                    )

                else:
                    static_graph = graph.to_static_csc()
                    sum_efeat = update_efeat_bipartite_e2e(
                        efeat, nfeat, static_graph, mode="sum"
                    )

        else:
            src_feat, dst_feat = nfeat, nfeat
            src, dst = (item.long() for item in graph.edges())
            sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

    else:
        src_feat, dst_feat = nfeat
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                src, dst = (item.long() for item in graph.to_dgl_graph().edges())
                sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                bipartite_graph = graph.to_bipartite_csc()
                sum_efeat = update_efeat_bipartite_e2e(
                    efeat, src_feat, dst_feat, bipartite_graph, mode="sum"
                )
        else:
            src, dst = (item.long() for item in graph.edges())
            sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

    return sum_efeat


@torch.jit.ignore()
def agg_concat_dgl(
    efeat: Tensor, dst_nfeat: Tensor, graph: DGLGraph, aggregation: str
) -> Tensor:
   
    with graph.local_scope():
        # populate features on graph edges
        graph.edata["x"] = efeat

        # aggregate edge features
        if aggregation == "sum":
            graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
        elif aggregation == "mean":
            graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
        else:
            raise RuntimeError("Not a valid aggregation!")

        # concat dst-node & edge features
        cat_feat = torch.cat((graph.dstdata["h_dest"], dst_nfeat), -1)
        return cat_feat


def aggregate_and_concat(
    efeat: Tensor,
    nfeat: Tensor,
    graph: Union[DGLGraph, CuGraphCSC],
    aggregation: str,
):

    if isinstance(graph, CuGraphCSC):
        # in this case, we don't have to distinguish a distributed setting
        # or the defalt setting as both efeat and nfeat are already
        # gurantueed to be on the same rank on both cases due to our
        # partitioning scheme

        if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
            cat_feat = agg_concat_dgl(efeat, nfeat, graph.to_dgl_graph(), aggregation)

        else:
            static_graph = graph.to_static_csc()
            cat_feat = agg_concat_e2n(nfeat, efeat, static_graph, aggregation)
    else:
        cat_feat = agg_concat_dgl(efeat, nfeat, graph, aggregation)

    return cat_feat


import functools
import logging
from typing import Tuple

import torch
from torch.autograd import Function

logger = logging.getLogger(__name__)

try:
    import nvfuser
    from nvfuser import DataType, FusionDefinition
except ImportError:
    logger.error(
        "An error occured. Either nvfuser is not installed or the version is "
        "incompatible. Please retry after installing correct version of nvfuser. "
        "The new version of nvfuser should be available in PyTorch container version "
        ">= 23.10. "
        "https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html. "
        "If using a source install method, please refer nvFuser repo for installation "
        "guidelines https://github.com/NVIDIA/Fuser.",
    )
    raise

_torch_dtype_to_nvfuser = {
    torch.double: DataType.Double,
    torch.float: DataType.Float,
    torch.half: DataType.Half,
    torch.int: DataType.Int,
    torch.int32: DataType.Int32,
    torch.bool: DataType.Bool,
    torch.bfloat16: DataType.BFloat16,
    torch.cfloat: DataType.ComplexFloat,
    torch.cdouble: DataType.ComplexDouble,
}


@functools.lru_cache(maxsize=None)
def silu_backward_for(
    fd: FusionDefinition,
    dtype: torch.dtype,
    dim: int,
    size: torch.Size,
    stride: Tuple[int, ...],
):  # pragma: no cover
    
    try:
        dtype = _torch_dtype_to_nvfuser[dtype]
    except KeyError:
        raise TypeError("Unsupported dtype")

    x = fd.define_tensor(
        shape=[-1] * dim,
        contiguity=nvfuser.compute_contiguity(size, stride),
        dtype=dtype,
    )
    one = fd.define_constant(1.0)

    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # z = sigmoid(x)
    grad_input = fd.ops.mul(y, fd.ops.add(one, fd.ops.mul(x, fd.ops.sub(one, y))))

    grad_input = fd.ops.cast(grad_input, dtype)

    fd.add_output(grad_input)


@functools.lru_cache(maxsize=None)
def silu_double_backward_for(
    fd: FusionDefinition,
    dtype: torch.dtype,
    dim: int,
    size: torch.Size,
    stride: Tuple[int, ...],
):  # pragma: no cover
   
    try:
        dtype = _torch_dtype_to_nvfuser[dtype]
    except KeyError:
        raise TypeError("Unsupported dtype")

    x = fd.define_tensor(
        shape=[-1] * dim,
        contiguity=nvfuser.compute_contiguity(size, stride),
        dtype=dtype,
    )
    one = fd.define_constant(1.0)

    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # dy = y * (1 - y)
    dy = fd.ops.mul(y, fd.ops.sub(one, y))
    # z = 1 + x * (1 - y)
    z = fd.ops.add(one, fd.ops.mul(x, fd.ops.sub(one, y)))
    # term1 = dy * z
    term1 = fd.ops.mul(dy, z)

    # term2 = y * ((1 - y) - x * dy)
    term2 = fd.ops.mul(y, fd.ops.sub(fd.ops.sub(one, y), fd.ops.mul(x, dy)))

    grad_input = fd.ops.add(term1, term2)

    grad_input = fd.ops.cast(grad_input, dtype)

    fd.add_output(grad_input)


@functools.lru_cache(maxsize=None)
def silu_triple_backward_for(
    fd: FusionDefinition,
    dtype: torch.dtype,
    dim: int,
    size: torch.Size,
    stride: Tuple[int, ...],
):  # pragma: no cover
   
    try:
        dtype = _torch_dtype_to_nvfuser[dtype]
    except KeyError:
        raise TypeError("Unsupported dtype")

    x = fd.define_tensor(
        shape=[-1] * dim,
        contiguity=nvfuser.compute_contiguity(size, stride),
        dtype=dtype,
    )
    one = fd.define_constant(1.0)
    two = fd.define_constant(2.0)

    # y = sigmoid(x)
    y = fd.ops.sigmoid(x)
    # dy = y * (1 - y)
    dy = fd.ops.mul(y, fd.ops.sub(one, y))
    # ddy = (1 - 2y) * dy
    ddy = fd.ops.mul(fd.ops.sub(one, fd.ops.mul(two, y)), dy)
    # term1 = ddy * (2 + x - 2xy)
    term1 = fd.ops.mul(
        ddy, fd.ops.sub(fd.ops.add(two, x), fd.ops.mul(two, fd.ops.mul(x, y)))
    )

    # term2 = dy * (1 - 2 (y + x * dy))
    term2 = fd.ops.mul(
        dy, fd.ops.sub(one, fd.ops.mul(two, fd.ops.add(y, fd.ops.mul(x, dy))))
    )

    grad_input = fd.ops.add(term1, term2)

    grad_input = fd.ops.cast(grad_input, dtype)

    fd.add_output(grad_input)


class FusedSiLU(Function):


    @staticmethod
    def forward(ctx, x):
    
        ctx.save_for_backward(x)
        return torch.nn.functional.silu(x)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
      
        (x,) = ctx.saved_tensors
        return FusedSiLU_deriv_1.apply(x) * grad_output


class FusedSiLU_deriv_1(Function):
  

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        with FusionDefinition() as fd:
            silu_backward_for(fd, x.dtype, x.dim(), x.size(), x.stride())
        out = fd.execute([x])[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        (x,) = ctx.saved_tensors
        return FusedSiLU_deriv_2.apply(x) * grad_output


class FusedSiLU_deriv_2(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        with FusionDefinition() as fd:
            silu_double_backward_for(fd, x.dtype, x.dim(), x.size(), x.stride())
        out = fd.execute([x])[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        (x,) = ctx.saved_tensors
        return FusedSiLU_deriv_3.apply(x) * grad_output


class FusedSiLU_deriv_3(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        with FusionDefinition() as fd:
            silu_triple_backward_for(fd, x.dtype, x.dim(), x.size(), x.stride())
        out = fd.execute([x])[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        (x,) = ctx.saved_tensors
        y = torch.sigmoid(x)
        dy = y * (1 - y)
        ddy = (1 - 2 * y) * dy
        dddy = (1 - 2 * y) * ddy - 2 * dy * dy
        z = 1 - 2 * (y + x * dy)
        term1 = dddy * (2 + x - 2 * x * y)
        term2 = 2 * ddy * z
        term3 = dy * (-2) * (2 * dy + x * ddy)
        return (term1 + term2 + term3) * grad_output


from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from torch import Tensor
from torch.autograd.function import once_differentiable


try:
    from transformer_engine import pytorch as te

    te_imported = True
except ImportError:
    te_imported = False


class CustomSiLuLinearAutogradFunction(torch.autograd.Function):
    """Custom SiLU + Linear autograd function"""

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
       
        out = F.silu(features)
        out = F.linear(out, weight, bias)
        ctx.save_for_backward(features, weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],]:
        """backward pass of the SiLU + Linear function"""

        from nvfuser import FusionDefinition

    

        (
            need_dgrad,
            need_wgrad,
            need_bgrad,
        ) = ctx.needs_input_grad
        features, weight = ctx.saved_tensors

        grad_features = None
        grad_weight = None
        grad_bias = None

        if need_bgrad:
            grad_bias = grad_output.sum(dim=0)

        if need_wgrad:
            out = F.silu(features)
            grad_weight = grad_output.T @ out

        if need_dgrad:
            grad_features = grad_output @ weight

            with FusionDefinition() as fd:
                silu_backward_for(
                    fd,
                    features.dtype,
                    features.dim(),
                    features.size(),
                    features.stride(),
                )

            grad_silu = fd.execute([features])[0]
            grad_features = grad_features * grad_silu

        return grad_features, grad_weight, grad_bias


class MeshGraphMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: Union[int, None] = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        if hidden_layers is not None:
            layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
            self.hidden_layers = hidden_layers
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
            layers.append(nn.Linear(hidden_dim, output_dim))

            self.norm_type = norm_type
            if norm_type is not None:
                if norm_type not in [
                    "LayerNorm",
                    "TELayerNorm",
                ]:
                    raise ValueError(
                        f"Invalid norm type {norm_type}. Supported types are LayerNorm and TELayerNorm."
                    )
                if norm_type == "TELayerNorm" and te_imported:
                    norm_layer = te.LayerNorm
                elif norm_type == "TELayerNorm" and not te_imported:
                    raise ValueError(
                        "TELayerNorm requires transformer-engine to be installed."
                    )
                else:
                    norm_layer = getattr(nn, norm_type)
                layers.append(norm_layer(output_dim))

            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

        if recompute_activation:
            if not isinstance(activation_fn, nn.SiLU):
                raise ValueError(activation_fn)
            self.recompute_activation = True
        else:
            self.recompute_activation = False

    def default_forward(self, x: Tensor) -> Tensor:
        """default forward pass of the MLP"""
        return self.model(x)

    @torch.jit.ignore()
    def custom_silu_linear_forward(self, x: Tensor) -> Tensor:
        """forward pass of the MLP where SiLU is recomputed in backward"""
        lin = self.model[0]
        hidden = lin(x)
        for i in range(1, self.hidden_layers + 1):
            lin = self.model[2 * i]
            hidden = CustomSiLuLinearAutogradFunction.apply(
                hidden, lin.weight, lin.bias
            )

        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers + 1]
            hidden = norm(hidden)
        return hidden

    def forward(self, x: Tensor) -> Tensor:
        if self.recompute_activation:
            return self.custom_silu_linear_forward(x)
        return self.default_forward(x)


class MeshGraphEdgeMLPConcat(MeshGraphMLP):
    def __init__(
        self,
        efeat_dim: int = 512,
        src_dim: int = 512,
        dst_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
        recompute_activation: bool = False,
    ):
        cat_dim = efeat_dim + src_dim + dst_dim
        super(MeshGraphEdgeMLPConcat, self).__init__(
            cat_dim,
            output_dim,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
            recompute_activation,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        efeat = concat_efeat(efeat, nfeat, graph)
        efeat = self.model(efeat)
        return efeat


class MeshGraphEdgeMLPSum(nn.Module):

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.efeat_dim = efeat_dim
        self.src_dim = src_dim
        self.dst_dim = dst_dim

        # this should ensure the same sequence of initializations
        # as the original MLP-Layer in combination with a concat operation
        tmp_lin = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim, bias=bias)
        # orig_weight has shape (hidden_dim, efeat_dim + src_dim + dst_dim)
        orig_weight = tmp_lin.weight
        w_efeat, w_src, w_dst = torch.split(
            orig_weight, [efeat_dim, src_dim, dst_dim], dim=1
        )
        self.lin_efeat = nn.Parameter(w_efeat)
        self.lin_src = nn.Parameter(w_src)
        self.lin_dst = nn.Parameter(w_dst)

        if bias:
            self.bias = tmp_lin.bias
        else:
            self.bias = None

        layers = [activation_fn]
        self.hidden_layers = hidden_layers
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.norm_type = norm_type
        if norm_type is not None:
            if norm_type not in [
                "LayerNorm",
                "TELayerNorm",
            ]:
                raise ValueError(
                    f"Invalid norm type {norm_type}. Supported types are LayerNorm and TELayerNorm."
                )
            if norm_type == "TELayerNorm" and te_imported:
                norm_layer = te.LayerNorm
            elif norm_type == "TELayerNorm" and not te_imported:
                raise ValueError(
                    "TELayerNorm requires transformer-engine to be installed."
                )
            else:
                norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))

        self.model = nn.Sequential(*layers)

        if recompute_activation:
            if not isinstance(activation_fn, nn.SiLU):
                raise ValueError(activation_fn)
            self.recompute_activation = True
        else:
            self.recompute_activation = False

    def forward_truncated_sum(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
    
        if isinstance(nfeat, Tensor):
            src_feat, dst_feat = nfeat, nfeat
        else:
            src_feat, dst_feat = nfeat
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = sum_efeat(mlp_efeat, (mlp_src, mlp_dst), graph)
        return mlp_sum

    def default_forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        """Default forward pass of the truncated MLP."""
        mlp_sum = self.forward_truncated_sum(
            efeat,
            nfeat,
            graph,
        )
        return self.model(mlp_sum)

    def custom_silu_linear_forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        """Forward pass of the truncated MLP with custom SiLU function."""
        mlp_sum = self.forward_truncated_sum(
            efeat,
            nfeat,
            graph,
        )
        lin = self.model[1]
        hidden = CustomSiLuLinearAutogradFunction.apply(mlp_sum, lin.weight, lin.bias)
        for i in range(2, self.hidden_layers + 1):
            lin = self.model[2 * i - 1]
            hidden = CustomSiLuLinearAutogradFunction.apply(
                hidden, lin.weight, lin.bias
            )

        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers]
            hidden = norm(hidden)
        return hidden

    def forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        if self.recompute_activation:
            return self.custom_silu_linear_forward(efeat, nfeat, graph)
        return self.default_forward(efeat, nfeat, graph)


from typing import Tuple

import torch.nn as nn
from torch import Tensor



class OneForecastEncoderEmbedder(nn.Module):

    def __init__(
        self,
        input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        # MLP for grid node embedding
        self.grid_node_mlp = MeshGraphMLP(
            input_dim=input_dim_grid_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for mesh node embedding
        self.mesh_node_mlp = MeshGraphMLP(
            input_dim=input_dim_mesh_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for mesh edge embedding
        self.mesh_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for grid2mesh edge embedding
        self.grid2mesh_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        g2m_efeat: Tensor,
        mesh_efeat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Input node feature embedding
        grid_nfeat = self.grid_node_mlp(grid_nfeat)
        mesh_nfeat = self.mesh_node_mlp(mesh_nfeat)
        # Input edge feature embedding
        g2m_efeat = self.grid2mesh_edge_mlp(g2m_efeat)
        mesh_efeat = self.mesh_edge_mlp(mesh_efeat)
        return grid_nfeat, mesh_nfeat, g2m_efeat, mesh_efeat


class OneForecastDecoderEmbedder(nn.Module):

    def __init__(
        self,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        # MLP for mesh2grid edge embedding
        self.mesh2grid_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self,
        m2g_efeat: Tensor,
    ) -> Tensor:
        m2g_efeat = self.mesh2grid_edge_mlp(m2g_efeat)
        return m2g_efeat



from typing import Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor



class MeshGraphDecoder(nn.Module):

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        # edge MLP
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # dst node MLP
        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        m2g_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        # update edge features
        efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat, grid_nfeat), graph)
        # aggregate messages (edge features) to obtain updated node features
        cat_feat = aggregate_and_concat(efeat, grid_nfeat, graph, self.aggregation)
        # transformation and residual connection
        dst_feat = self.node_mlp(cat_feat) + grid_nfeat
        return dst_feat



from typing import Tuple, Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor




class MeshGraphEncoder(nn.Module):

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_src_nodes: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: int = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        # edge MLP
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # src node MLP
        self.src_node_mlp = MeshGraphMLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # dst node MLP
        self.dst_node_mlp = MeshGraphMLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        g2m_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tuple[Tensor, Tensor]:
        # update edge features by concatenating node features (both mesh and grid) and existing edge featues
        # (or applying the concat trick instead)
        efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
        # aggregate messages (edge features) to obtain updated node features
        cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
        # update src, dst node features + residual connections
        mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
        return grid_nfeat, mesh_nfeat




import torch
import torch.nn as nn


Tensor = torch.Tensor


class Identity(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x


class Stan(nn.Module):

    def __init__(self, out_features: int = 1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.beta.shape[-1]:
            raise ValueError(
                f"The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}"
            )
        return torch.tanh(x) * (1.0 + self.beta * x)


class SquarePlus(nn.Module):

    def __init__(self):
        super().__init__()
        self.b = 4

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (x + torch.sqrt(x * x + self.b))


class CappedLeakyReLU(torch.nn.Module):
   

    def __init__(self, cap_value=1.0, **kwargs):
       
        super().__init__()
        self.add_module("leaky_relu", torch.nn.LeakyReLU(**kwargs))
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.leaky_relu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x


class CappedGELU(torch.nn.Module):
   

    def __init__(self, cap_value=1.0, **kwargs):
       

        super().__init__()
        self.add_module("gelu", torch.nn.GELU(**kwargs))
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x


# Dictionary of activation functions
ACT2FN = {
    "relu": nn.ReLU,
    "leaky_relu": (nn.LeakyReLU, {"negative_slope": 0.1}),
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "logsigmoid": nn.LogSigmoid,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": (nn.Threshold, {"threshold": 1.0, "value": 1.0}),
    "hardtanh": nn.Hardtanh,
    "identity": Identity,
    "stan": Stan,
    "squareplus": SquarePlus,
    "cappek_leaky_relu": CappedLeakyReLU,
    "capped_gelu": CappedGELU,
}


def get_activation(activation: str) -> nn.Module:
   
    try:
        activation = activation.lower()
        module = ACT2FN[activation]
        if isinstance(module, tuple):
            return module[0](**module[1])
        else:
            return module()
    except KeyError:
        raise KeyError(
            f"Activation function {activation} not found. Available options are: {list(ACT2FN.keys())}"
        )



from dataclasses import dataclass


@dataclass
class ModelMetaData:
    """Data class for storing essential meta data needed for all Modulus Models"""

    # Model info
    name: str = "ModulusModule"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    amp_cpu: bool = None
    amp_gpu: bool = None
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    onnx_gpu: bool = None
    onnx_cpu: bool = None
    onnx_runtime: bool = False
    trt: bool = False
    # Physics informed
    var_dim: int = -1
    func_torch: bool = False
    auto_grad: bool = False

    def __post_init__(self):
        self.amp_cpu = self.amp if self.amp_cpu is None else self.amp_cpu
        self.amp_gpu = self.amp if self.amp_gpu is None else self.amp_gpu
        self.onnx_cpu = self.onnx if self.onnx_cpu is None else self.onnx_cpu
        self.onnx_gpu = self.onnx if self.onnx_gpu is None else self.onnx_gpu



from importlib.metadata import EntryPoint, entry_points
from typing import List, Union

# This import is required for compatibility with doctests.
import importlib_metadata


class ModelRegistry:
    _shared_state = {"_model_registry": None}

    def __new__(cls, *args, **kwargs):
        obj = super(ModelRegistry, cls).__new__(cls)
        obj.__dict__ = cls._shared_state
        if cls._shared_state["_model_registry"] is None:
            cls._shared_state["_model_registry"] = cls._construct_registry()
        return obj

    @staticmethod
    def _construct_registry() -> dict:
        registry = {}
        entrypoints = entry_points(group="modulus.models")
        for entry_point in entrypoints:
            registry[entry_point.name] = entry_point
        return registry

    def register(self, model: "modulus.Module", name: Union[str, None] = None) -> None:
       

        # Check if model is a modulus model
        if not issubclass(model, modulus.Module):
            raise ValueError(
                f"Only subclasses of modulus.Module can be registered. "
                f"Provided model is of type {type(model)}"
            )

        # If no name provided, use the model's name
        if name is None:
            name = model.__name__

        # Check if name already in use
        if name in self._model_registry:
            raise ValueError(f"Name {name} already in use")

        # Add this class to the dict of model registry
        self._model_registry[name] = model

    def factory(self, name: str) -> "modulus.Module":
       
        model = self._model_registry.get(name)
        if model is not None:
            if isinstance(model, (EntryPoint, importlib_metadata.EntryPoint)):
                model = model.load()
            return model

        raise KeyError(f"No model is registered under the name {name}")

    def list_models(self) -> List[str]:
       
        return list(self._model_registry.keys())

    def __clear_registry__(self):
        # NOTE: This is only used for testing purposes
        self._model_registry = {}

    def __restore_registry__(self):
        # NOTE: This is only used for testing purposes
        self._model_registry = self._construct_registry()


import importlib
import inspect
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import torch



class Module(torch.nn.Module):
   

    _file_extension = ".mdlus"  # Set file extension for saving and loading
    __model_checkpoint_version__ = (
        "0.1.0"  # Used for file versioning and is not the same as modulus version
    )

    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls)

        # Get signature of __init__ function
        sig = inspect.signature(cls.__init__)

        # Bind args and kwargs to signature
        bound_args = sig.bind_partial(
            *([None] + list(args)), **kwargs
        )  # Add None to account for self
        bound_args.apply_defaults()

        # Get args and kwargs (excluding self and unroll kwargs)
        instantiate_args = {}
        for param, (k, v) in zip(sig.parameters.values(), bound_args.arguments.items()):
            # Skip self
            if k == "self":
                continue

            # Add args and kwargs to instantiate_args
            if param.kind == param.VAR_KEYWORD:
                instantiate_args.update(v)
            else:
                instantiate_args[k] = v

        # Store args needed for instantiation
        out._args = {
            "__name__": cls.__name__,
            "__module__": cls.__module__,
            "__args__": instantiate_args,
        }
        return out

    def __init__(self, meta: Union[ModelMetaData, None] = None):
        super().__init__()
        self.meta = meta
        self.register_buffer("device_buffer", torch.empty(0))
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("core.module")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    @staticmethod
    def _safe_members(tar, local_path):
        for member in tar.getmembers():
            if (
                ".." in member.name
                or os.path.isabs(member.name)
                or os.path.realpath(os.path.join(local_path, member.name)).startswith(
                    os.path.realpath(local_path)
                )
            ):
                yield member
            else:
                print(f"Skipping potentially malicious file: {member.name}")

    @classmethod
    def instantiate(cls, arg_dict: Dict[str, Any]) -> "Module":
       

        _cls_name = arg_dict["__name__"]
        registry = ModelRegistry()
        if cls.__name__ == arg_dict["__name__"]:  # If cls is the class
            _cls = cls
        elif _cls_name in registry.list_models():  # Built in registry
            _cls = registry.factory(_cls_name)
        else:
            try:
                # Otherwise, try to import the class
                _mod = importlib.import_module(arg_dict["__module__"])
                _cls = getattr(_mod, arg_dict["__name__"])
            except AttributeError:
                # Cross fingers and hope for the best (maybe the class name changed)
                _cls = cls
        return _cls(**arg_dict["__args__"])

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s - {self.meta.name}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # TODO: set up debug log
        # fh = logging.FileHandler(f'modulus-core-{self.meta.name}.log')

    def save(self, file_name: Union[str, None] = None, verbose: bool = False) -> None:
       

        if file_name is not None and not file_name.endswith(self._file_extension):
            raise ValueError(
                f"File name must end with {self._file_extension} extension"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            torch.save(self.state_dict(), local_path / "model.pt")

            with open(local_path / "args.json", "w") as f:
                json.dump(self._args, f)

            # Save the modulus version and git hash (if available)
            metadata_info = {
                "modulus_version": modulus.__version__,
                "mdlus_file_version": self.__model_checkpoint_version__,
            }

            if verbose:
                import git

                try:
                    repo = git.Repo(search_parent_directories=True)
                    metadata_info["git_hash"] = repo.head.object.hexsha
                except git.InvalidGitRepositoryError:
                    metadata_info["git_hash"] = None

            with open(local_path / "metadata.json", "w") as f:
                json.dump(metadata_info, f)

            # Once all files are saved, package them into a tar file
            with tarfile.open(local_path / "model.tar", "w") as tar:
                for file in local_path.iterdir():
                    tar.add(str(file), arcname=file.name)

            if file_name is None:
                file_name = self.meta.name + ".mdlus"

            # Save files to remote destination
            fs = _get_fs(file_name)
            fs.put(str(local_path / "model.tar"), file_name)

    @staticmethod
    def _check_checkpoint(local_path: str) -> bool:
        if not local_path.joinpath("args.json").exists():
            raise IOError("File 'args.json' not found in checkpoint")

        if not local_path.joinpath("metadata.json").exists():
            raise IOError("File 'metadata.json' not found in checkpoint")

        if not local_path.joinpath("model.pt").exists():
            raise IOError("Model weights 'model.pt' not found in checkpoint")

        # Check if the checkpoint version is compatible with the current version
        with open(local_path.joinpath("metadata.json"), "r") as f:
            metadata_info = json.load(f)
            if (
                metadata_info["mdlus_file_version"]
                != Module.__model_checkpoint_version__
            ):
                raise IOError(
                    f"Model checkpoint version {metadata_info['mdlus_file_version']} is not compatible with current version {Module.__version__}"
                )

    def load(
        self,
        file_name: str,
        map_location: Union[None, str, torch.device] = None,
        strict: bool = True,
    ) -> None:
       

        # Download and cache the checkpoint file if needed
        cached_file_name = _download_cached(file_name)

        # Use a temporary directory to extract the tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            # Open the tar file and extract its contents to the temporary directory
            with tarfile.open(cached_file_name, "r") as tar:
                tar.extractall(
                    path=local_path, members=list(Module._safe_members(tar, local_path))
                )

            # Check if the checkpoint is valid
            Module._check_checkpoint(local_path)

            # Load the model weights
            device = map_location if map_location is not None else self.device
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=device
            )
            self.load_state_dict(model_dict, strict=strict)

    @classmethod
    def from_checkpoint(cls, file_name: str) -> "Module":
       
        # Download and cache the checkpoint file if needed
        cached_file_name = _download_cached(file_name)

        # Use a temporary directory to extract the tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            # Open the tar file and extract its contents to the temporary directory
            with tarfile.open(cached_file_name, "r") as tar:
                tar.extractall(
                    path=local_path, members=list(cls._safe_members(tar, local_path))
                )

            # Check if the checkpoint is valid
            Module._check_checkpoint(local_path)

            # Load model arguments and instantiate the model
            with open(local_path.joinpath("args.json"), "r") as f:
                args = json.load(f)
            model = cls.instantiate(args)

            # Load the model weights
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=model.device
            )
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def from_torch(
        torch_model_class: torch.nn.Module, meta: ModelMetaData = None
    ) -> "Module":
       

        # Define an internal class as before
        class ModulusModel(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(meta=meta)
                self.inner_model = torch_model_class(*args, **kwargs)

            def forward(self, x):
                return self.inner_model(x)

        # Get the argument names and default values of the PyTorch model's init method
        init_argspec = inspect.getfullargspec(torch_model_class.__init__)
        model_argnames = init_argspec.args[1:]  # Exclude 'self'
        model_defaults = init_argspec.defaults or []
        defaults_dict = dict(
            zip(model_argnames[-len(model_defaults) :], model_defaults)
        )

        # Define the signature of new init
        params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        params += [
            inspect.Parameter(
                argname,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults_dict.get(argname, inspect.Parameter.empty),
            )
            for argname in model_argnames
        ]
        init_signature = inspect.Signature(params)

        # Replace ModulusModel.__init__ signature with new init signature
        ModulusModel.__init__.__signature__ = init_signature

        # Generate a unique name for the created class
        new_class_name = f"{torch_model_class.__name__}ModulusModel"
        ModulusModel.__name__ = new_class_name

        # Add this class to the dict of models classes
        registry = ModelRegistry()
        registry.register(ModulusModel, new_class_name)

        return ModulusModel

    @property
    def device(self) -> torch.device:
        
        return self.device_buffer.device

    def num_parameters(self) -> int:
        """Gets the number of learnable parameters"""
        count = 0
        for name, param in self.named_parameters():
            count += param.numel()
        return count



from typing import List, Tuple

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from torch import Tensor, testing


def create_graph(
    src: List,
    dst: List,
    to_bidirected: bool = True,
    add_self_loop: bool = False,
    dtype: torch.dtype = torch.int32,
) -> DGLGraph:
   
    graph = dgl.graph((src, dst), idtype=dtype)
    if to_bidirected:
        graph = dgl.to_bidirected(graph)
    if add_self_loop:
        graph = dgl.add_self_loop(graph)
    return graph


def create_heterograph(
    src: List,
    dst: List,
    labels: str,
    dtype: torch.dtype = torch.int32,
    num_nodes_dict: dict = None,
) -> DGLGraph:
   
    graph = dgl.heterograph(
        {labels: ("coo", (src, dst))}, num_nodes_dict=num_nodes_dict, idtype=dtype
    )
    return graph


def add_edge_features(graph: DGLGraph, pos: Tensor, normalize: bool = True) -> DGLGraph:
   

    if isinstance(pos, tuple):
        src_pos, dst_pos = pos
    else:
        src_pos = dst_pos = pos
    src, dst = graph.edges()

    src_pos, dst_pos = src_pos[src.long()], dst_pos[dst.long()]
    dst_latlon = xyz2latlon(dst_pos, unit="rad")
    dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]

    # azimuthal & polar rotation
    theta_azimuthal = azimuthal_angle(dst_lon)
    theta_polar = polar_angle(dst_lat)

    src_pos = geospatial_rotation(src_pos, theta=theta_azimuthal, axis="z", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_azimuthal, axis="z", unit="rad")
    # y values should be zero
    try:
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")
    src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y", unit="rad")
    # x values should be one, y & z values should be zero
    try:
        testing.assert_close(dst_pos[:, 0], torch.ones_like(dst_pos[:, 0]))
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
        testing.assert_close(dst_pos[:, 2], torch.zeros_like(dst_pos[:, 2]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")

    # prepare edge features
    disp = src_pos - dst_pos
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

    # normalize using the longest edge
    if normalize:
        max_disp_norm = torch.max(disp_norm)
        graph.edata["x"] = torch.cat(
            (disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1
        )
    else:
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)
    return graph


def add_node_features(graph: DGLGraph, pos: Tensor) -> DGLGraph:
  
    latlon = xyz2latlon(pos)
    lat, lon = latlon[:, 0], latlon[:, 1]
    graph.ndata["x"] = torch.stack(
        (torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1
    )
    return graph


def latlon2xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
   
    if unit == "deg":
        latlon = deg2rad(latlon)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


def xyz2latlon(xyz: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == "deg":
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    elif unit == "rad":
        return torch.stack((lat, lon), dim=1)
    else:
        raise ValueError("Not a valid unit")


def geospatial_rotation(
    invar: Tensor, theta: Tensor, axis: str, unit: str = "rad"
) -> Tensor:
    

    # get the right unit
    if unit == "deg":
        invar = rad2deg(invar)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")

    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3))
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    if axis == "x":
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == "y":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == "z":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError("Invalid axis")

    outvar = torch.matmul(rotation, invar)
    outvar = outvar.squeeze()
    return outvar


def azimuthal_angle(lon: Tensor) -> Tensor:
   
    angle = torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)
    return angle


def polar_angle(lat: Tensor) -> Tensor:
   
    angle = torch.where(lat >= 0.0, lat, 2 * np.pi + lat)
    return angle


def deg2rad(deg: Tensor) -> Tensor:
   
    return deg * np.pi / 180


def rad2deg(rad):
    
    return rad * 180 / np.pi


def cell_to_adj(cells: List[List[int]]):
   
    num_cells = np.shape(cells)[0]
    src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]
    dst = [cells[i][indx] for i in range(num_cells) for indx in [1, 2, 0]]
    return src, dst


def max_edge_length(
    vertices: List[List[float]], source_nodes: List[int], destination_nodes: List[int]
) -> float:
    
    vertices_np = np.array(vertices)
    source_coords = vertices_np[source_nodes]
    dest_coords = vertices_np[destination_nodes]

    # Compute the squared distances for all edges
    squared_differences = np.sum((source_coords - dest_coords) ** 2, axis=1)

    # Compute the maximum edge length
    max_length = np.sqrt(np.max(squared_differences))

    return max_length


def get_face_centroids(
    vertices: List[Tuple[float, float, float]], faces: List[List[int]]
) -> List[Tuple[float, float, float]]:
    
    centroids = []

    for face in faces:
        # Extract the coordinates of the vertices for the current face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Compute the centroid of the triangle
        centroid = (
            (v0[0] + v1[0] + v2[0]) / 3,
            (v0[1] + v1[1] + v2[1]) / 3,
            (v0[2] + v1[2] + v2[2]) / 3,
        )

        centroids.append(centroid)

    return centroids


import itertools
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.spatial import transform


class TriangularMesh(NamedTuple):
   

    vertices: np.ndarray
    faces: np.ndarray


def merge_meshes(mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
   
    for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

    return TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
    )


def get_hierarchy_of_triangular_meshes_for_sphere(splits: int) -> List[TriangularMesh]:
    
    current_mesh = get_icosahedron()
    output_meshes = [current_mesh]
    for _ in range(splits):
        current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
        output_meshes.append(current_mesh)
    return output_meshes


def get_icosahedron() -> TriangularMesh:
    
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in [1.0, -1.0]:
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.0))
            vertices.append((0.0, c1, c2))
            vertices.append((c2, 0.0, c1))

    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1.0, phi])

    # I did this manually, checking the orientation one by one.
    faces = [
        (0, 1, 2),
        (0, 6, 1),
        (8, 0, 2),
        (8, 4, 0),
        (3, 8, 2),
        (3, 2, 7),
        (7, 2, 1),
        (0, 4, 6),
        (4, 11, 6),
        (6, 11, 5),
        (1, 5, 7),
        (4, 10, 11),
        (4, 8, 10),
        (10, 8, 3),
        (10, 3, 9),
        (11, 10, 9),
        (11, 9, 5),
        (5, 9, 7),
        (9, 3, 7),
        (1, 6, 5),
    ]

    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    rotation = transform.Rotation.from_euler(seq="y", angles=rotation_angle)
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)

    return TriangularMesh(
        vertices=vertices.astype(np.float32), faces=np.array(faces, dtype=np.int32)
    )


def _two_split_unit_sphere_triangle_faces(
    triangular_mesh: TriangularMesh,
) -> TriangularMesh:
    
   
    new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

    new_faces = []
    for ind1, ind2, ind3 in triangular_mesh.faces:
      
        ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
       
        new_faces.extend(
            [
                [ind1, ind12, ind31],  # 1
                [ind12, ind2, ind23],  # 2
                [ind31, ind23, ind3],  # 3
                [ind12, ind23, ind31],  # 4
            ]
        )
    return TriangularMesh(
        vertices=new_vertices_builder.get_all_vertices(),
        faces=np.array(new_faces, dtype=np.int32),
    )
from sklearn.neighbors import NearestNeighbors

def add_selected_pairs_for_latlon(
    mesh: TriangularMesh,
    latlon_pairs: List[Tuple[float, float, float, float]],
    unit: str = "deg"
) -> TriangularMesh:
   
    vertices_xyz = torch.from_numpy(mesh.vertices.astype(np.float32))  # [N,3]
    vertex_latlon = xyz2latlon(vertices_xyz, unit="deg")  # [N,2] 默认用 deg
    if unit == "rad":
        factor = 180.0 / np.pi
        vertex_latlon = vertex_latlon * factor

    vertex_latlon_np = vertex_latlon.numpy()  # [N,2]

    nbrs = NearestNeighbors(n_neighbors=1).fit(vertex_latlon_np)

    new_faces = []
    for (lat1, lon1, lat2, lon2) in latlon_pairs:
        if unit == "rad":
            lat1 = lat1 * 180.0 / np.pi
            lon1 = lon1 * 180.0 / np.pi
            lat2 = lat2 * 180.0 / np.pi
            lon2 = lon2 * 180.0 / np.pi

        query1 = np.array([[lat1, lon1]], dtype=np.float32)  # [1,2]
        dist1, idx1 = nbrs.kneighbors(query1)
        idx1 = idx1[0, 0]

        query2 = np.array([[lat2, lon2]], dtype=np.float32)  # [1,2]
        dist2, idx2 = nbrs.kneighbors(query2)
        idx2 = idx2[0, 0]

        new_faces.append([idx1, idx2, idx1])

    if not new_faces:
        return mesh  

    faces_combined = np.concatenate([mesh.faces, np.array(new_faces, dtype=np.int32)], axis=0)
    return TriangularMesh(mesh.vertices, faces_combined)

def local_refine_sphere_triangles(
    mesh: TriangularMesh,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    extra_splits: int = 2,
    complex_edge = False,
    latlon_pairs: Optional[List[Tuple[float, float, float, float]]] = None,
) -> TriangularMesh:

    vertices = mesh.vertices
    faces = mesh.faces  # (F,3)

    centroids = get_face_centroids(vertices, faces)  # (F,3)
    centroids = np.array(centroids, dtype=np.float32)
    centroids_torch = torch.from_numpy(centroids)
    latlon = xyz2latlon(centroids_torch, unit="deg")  # (F,2)
    lat, lon = latlon[:, 0], latlon[:, 1]

    mask_local = (
        (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )

    faces_local = faces[ mask_local.numpy() ]
    faces_other = faces[~mask_local.numpy() ]

    mesh_local = TriangularMesh(vertices=vertices, faces=faces_local)
    for _ in range(extra_splits):
        mesh_local = _two_split_unit_sphere_triangle_faces(mesh_local)

    mesh_other = TriangularMesh(vertices=vertices, faces=faces_other)

    refined = merge_two_meshes(mesh_local, mesh_other)
    refined = remove_duplicate_vertices(refined)

    if complex_edge and latlon_pairs is not None and len(latlon_pairs) > 0:
        refined = add_selected_pairs_for_latlon(refined, latlon_pairs, unit="deg")
        refined = remove_duplicate_vertices(refined)

    return refined


def merge_two_meshes(m1: TriangularMesh, m2: TriangularMesh) -> TriangularMesh:
    v1, f1 = m1.vertices, m1.faces
    v2, f2 = m2.vertices, m2.faces
    offset = v1.shape[0]
    f2_shifted = f2 + offset
    merged_vertices = np.concatenate([v1, v2], axis=0)
    merged_faces = np.concatenate([f1, f2_shifted], axis=0)
    return TriangularMesh(merged_vertices, merged_faces)


def remove_duplicate_vertices(mesh: TriangularMesh) -> TriangularMesh:
    vertices = mesh.vertices
    faces = mesh.faces
    rounded = np.round(vertices, decimals=6)
    dct = {}
    idx_map = np.zeros(vertices.shape[0], dtype=np.int64)
    new_vertices_list = []
    new_idx = 0

    for old_i, coords in enumerate(rounded):
        key = tuple(coords.tolist())
        if key not in dct:
            dct[key] = new_idx
            idx_map[old_i] = new_idx
            new_vertices_list.append(vertices[old_i])
            new_idx += 1
        else:
            idx_map[old_i] = dct[key]

    new_vertices = np.array(new_vertices_list, dtype=np.float32)
    new_faces = idx_map[faces]
    return TriangularMesh(new_vertices, new_faces)



class _ChildVerticesBuilder(object):
    """Bookkeeping of new child vertices added to an existing set of vertices."""

    def __init__(self, parent_vertices):

       
        self._child_vertices_index_mapping = {}
        self._parent_vertices = parent_vertices
       
        self._all_vertices_list = list(parent_vertices)

    def _get_child_vertex_key(self, parent_vertex_indices):
        return tuple(sorted(parent_vertex_indices))

    def _create_child_vertex(self, parent_vertex_indices):
        
        child_vertex_position = self._parent_vertices[list(parent_vertex_indices)].mean(
            0
        )
        child_vertex_position /= np.linalg.norm(child_vertex_position)

        
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        self._child_vertices_index_mapping[child_vertex_key] = len(
            self._all_vertices_list
        )
        self._all_vertices_list.append(child_vertex_position)

    def get_new_child_vertex_index(self, parent_vertex_indices):
        
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        if child_vertex_key not in self._child_vertices_index_mapping:
            self._create_child_vertex(parent_vertex_indices)
        return self._child_vertices_index_mapping[child_vertex_key]

    def get_all_vertices(self):
        return np.array(self._all_vertices_list)


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    assert faces.ndim == 2
    assert faces.shape[-1] == 3
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


import logging

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor



logger = logging.getLogger(__name__)


class Graph:

    def __init__(
        self,
        lat_lon_grid: Tensor,
        mesh_level: int = 6,
        multimesh: bool = True,
        khop_neighbors: int = 0,
        dtype=torch.float,
        do_local_refine: bool = False,
        refine_specs: Optional[List[Tuple[float, float, float, float, int]]] = None,
        complex_edge: bool = False,
        lon_lat_complex_edge: Optional[List[Tuple[float, float, float, float]]] = None

    ) -> None:
        self.khop_neighbors = khop_neighbors
        self.dtype = dtype
        self.do_local_refine = do_local_refine
        self.refine_specs = refine_specs
        self.complex_edge = complex_edge
        self.lon_lat_complex_edge = lon_lat_complex_edge

        # flatten lat/lon gird
        self.lat_lon_grid_flat = lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)


        _meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
        finest_mesh = _meshes[-1]
        self.finest_mesh_src, self.finest_mesh_dst = faces_to_edges(finest_mesh.faces)
        self.finest_mesh_vertices = np.array(finest_mesh.vertices)

    
        if self.do_local_refine:
            mesh_local = finest_mesh
            for (lat_min, lat_max, lon_min, lon_max, extra_splits) in self.refine_specs:
                mesh_local = local_refine_sphere_triangles(
                    mesh=mesh_local,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    extra_splits=extra_splits,
                    complex_edge=complex_edge,
                    latlon_pairs=lon_lat_complex_edge
                )
            refined_finest = mesh_local

        if multimesh:
            if self.do_local_refine:
                mesh = merge_meshes(_meshes + [refined_finest])
                self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)
                self.mesh_vertices = np.array(mesh.vertices)
            else:
                if complex_edge and lon_lat_complex_edge is not None and len(lon_lat_complex_edge) > 0:
                    refined = add_selected_pairs_for_latlon(refined, lon_lat_complex_edge, unit="deg")
                    refined = remove_duplicate_vertices(refined)
                    mesh = merge_meshes(_meshes + [refined_finest])
                    self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)
                    self.mesh_vertices = np.array(mesh.vertices)
                else:
                    mesh = merge_meshes(_meshes)
                    self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)
                    self.mesh_vertices = np.array(mesh.vertices)

        else:
            mesh = refined_finest
            self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)
            self.mesh_vertices = np.array(mesh.vertices)

        self.mesh_faces = mesh.faces

    @staticmethod
    def khop_adj_all_k(g, kmax):
        if not g.is_homogeneous:
            raise NotImplementedError("only homogeneous graph is supported")
        min_degree = g.in_degrees().min()
        with torch.no_grad():
            adj = g.adj_external(transpose=True, scipy_fmt=None)
            adj_k = adj
            adj_all = adj.clone()
            for _ in range(2, kmax + 1):
                
                adj_k = (adj @ adj_k) / min_degree
                adj_all += adj_k
        return adj_all.to_dense().bool()

    def create_mesh_graph(self, verbose: bool = True) -> Tensor:
        
        mesh_graph = create_graph(
            self.mesh_src,
            self.mesh_dst,
            to_bidirected=True,
            add_self_loop=False,
            dtype=torch.int32,
        )
        mesh_pos = torch.tensor(
            self.mesh_vertices,
            dtype=torch.float32,
        )
        mesh_graph = add_edge_features(mesh_graph, mesh_pos)
        mesh_graph = add_node_features(mesh_graph, mesh_pos)
        mesh_graph.ndata["lat_lon"] = xyz2latlon(mesh_pos)
        # ensure fields set to dtype to avoid later conversions
        mesh_graph.ndata["x"] = mesh_graph.ndata["x"].to(dtype=self.dtype)
        mesh_graph.edata["x"] = mesh_graph.edata["x"].to(dtype=self.dtype)
        if self.khop_neighbors > 0:
            khop_adj_bool = self.khop_adj_all_k(g=mesh_graph, kmax=self.khop_neighbors)
            mask = ~khop_adj_bool
        else:
            mask = None
        if verbose:
            print("mesh graph:", mesh_graph)
        return mesh_graph, mask

    def create_g2m_graph(self, verbose: bool = True) -> Tensor:
      

        max_edge_len = max_edge_length(
            self.finest_mesh_vertices, self.finest_mesh_src, self.finest_mesh_dst
        )

        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 4
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(self.mesh_vertices)
        distances, indices = neighbors.kneighbors(cartesian_grid)

        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(n_nbrs):
                if distances[i][j] <= 0.6 * max_edge_len:
                    src.append(i)
                    dst.append(indices[i][j])

        g2m_graph = create_heterograph(
            src, dst, ("grid", "g2m", "mesh"), dtype=torch.int32
        )
        g2m_graph.srcdata["pos"] = cartesian_grid.to(torch.float32)
        g2m_graph.dstdata["pos"] = torch.tensor(
            self.mesh_vertices,
            dtype=torch.float32,
        )
        g2m_graph.srcdata["lat_lon"] = self.lat_lon_grid_flat
        g2m_graph.dstdata["lat_lon"] = xyz2latlon(g2m_graph.dstdata["pos"])

        g2m_graph = add_edge_features(
            g2m_graph, (g2m_graph.srcdata["pos"], g2m_graph.dstdata["pos"])
        )

        # avoid potential conversions at later points
        g2m_graph.srcdata["pos"] = g2m_graph.srcdata["pos"].to(dtype=self.dtype)
        g2m_graph.dstdata["pos"] = g2m_graph.dstdata["pos"].to(dtype=self.dtype)
        g2m_graph.ndata["pos"]["grid"] = g2m_graph.ndata["pos"]["grid"].to(
            dtype=self.dtype
        )
        g2m_graph.ndata["pos"]["mesh"] = g2m_graph.ndata["pos"]["mesh"].to(
            dtype=self.dtype
        )
        g2m_graph.edata["x"] = g2m_graph.edata["x"].to(dtype=self.dtype)
        if verbose:
            print("g2m graph:", g2m_graph)
        return g2m_graph

    def create_m2g_graph(self, verbose: bool = True) -> Tensor:
        
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        face_centroids = get_face_centroids(self.mesh_vertices, self.mesh_faces)
        n_nbrs = 1
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(face_centroids)
        _, indices = neighbors.kneighbors(cartesian_grid)
        indices = indices.flatten()

        src = [p for i in indices for p in self.mesh_faces[i]]
        dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]
        m2g_graph = create_heterograph(
            src, dst, ("mesh", "m2g", "grid"), dtype=torch.int32
        )  # number of edges is 3,114,720, exactly matches with the paper

        m2g_graph.srcdata["pos"] = torch.tensor(
            self.mesh_vertices,
            dtype=torch.float32,
        )
        m2g_graph.dstdata["pos"] = cartesian_grid.to(dtype=torch.float32)

        m2g_graph.srcdata["lat_lon"] = xyz2latlon(m2g_graph.srcdata["pos"])
        m2g_graph.dstdata["lat_lon"] = self.lat_lon_grid_flat

        m2g_graph = add_edge_features(
            m2g_graph, (m2g_graph.srcdata["pos"], m2g_graph.dstdata["pos"])
        )
        # avoid potential conversions at later points
        m2g_graph.srcdata["pos"] = m2g_graph.srcdata["pos"].to(dtype=self.dtype)
        m2g_graph.dstdata["pos"] = m2g_graph.dstdata["pos"].to(dtype=self.dtype)
        m2g_graph.ndata["pos"]["grid"] = m2g_graph.ndata["pos"]["grid"].to(
            dtype=self.dtype
        )
        m2g_graph.ndata["pos"]["mesh"] = m2g_graph.ndata["pos"]["mesh"].to(
            dtype=self.dtype
        )
        m2g_graph.edata["x"] = m2g_graph.edata["x"].to(dtype=self.dtype)

        if verbose:
            print("m2g graph:", m2g_graph)
        return m2g_graph



from typing import Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

class MeshEdgeBlockMultiHeadGated(nn.Module):

    def __init__(
        self,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
        num_heads: int = 4, 
    ):
        super().__init__()
        self.num_heads = num_heads

        if do_concat_trick:
            MLP = MeshGraphEdgeMLPSum
        else:
            MLP = MeshGraphEdgeMLPConcat

        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

       
        gating_hidden = max(16, hidden_dim // 8)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim_edges + input_dim_nodes*2, gating_hidden),
            activation_fn,
            nn.Linear(gating_hidden, 3 * num_heads),
            nn.Sigmoid(),
        )

        self.output_dim = output_dim

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, "object"],
    ) -> Tuple[Tensor, Tensor]:

        src_idx, dst_idx = graph.edges()
        src_feat_edge = nfeat[src_idx]
        dst_feat_edge = nfeat[dst_idx]
        cat_raw = torch.cat([efeat, src_feat_edge, dst_feat_edge], dim=-1)
        gating_all = self.gate_net(cat_raw)  # shape=[#edges, 3*num_heads]

        gating_all = gating_all.view(-1, self.num_heads, 3)
   
        gating = gating_all.mean(dim=1)

        g_e = gating[:, 0:1]
        g_s = gating[:, 1:2]
        g_d = gating[:, 2:3]

        efeat_updated = self.edge_mlp(efeat, nfeat, graph)

        efeat_new = efeat_updated * (g_e + g_s + g_d) / 3.0 + efeat
        return efeat_new, nfeat

from typing import Tuple, Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

class MeshNodeBlockMultiHeadAttn(nn.Module):
   
    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.aggregation = aggregation

        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + input_dim_edges * num_heads, 
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        attn_hidden = max(16, hidden_dim // 8)
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim_edges, attn_hidden),
            activation_fn,
            nn.Linear(attn_hidden, num_heads),  
        )

    def forward(
        self,
        efeat: Tensor,  # [#edges, input_dim_edges]
        nfeat: Tensor,  # [#nodes, input_dim_nodes]
        graph: Union[DGLGraph, "object"],
    ) -> Tuple[Tensor, Tensor]:

        attn_logits = self.attn_net(efeat)  

        with graph.local_scope():
            graph.edata["logits"] = attn_logits
            graph.edata["score"] = dgl.nn.functional.edge_softmax(graph, attn_logits)
            alpha = graph.edata["score"]  

        efeat_heads = efeat.unsqueeze(1) 
        efeat_heads = efeat_heads.expand(-1, self.num_heads, -1)
        weighted_efeat_heads = efeat_heads * alpha.unsqueeze(-1)

        with graph.local_scope():
            graph.edata["x"] = weighted_efeat_heads
            def reduce_func(nodes):
               
                return {"h_dest": nodes.mailbox["m"].sum(dim=1)}
            graph.update_all(
                message_func=fn.copy_e("x", "m"),
                reduce_func=reduce_func,
            )
            dst_all = graph.dstdata["h_dest"]

        dst_all = dst_all.view(dst_all.shape[0], -1)

        cat_feat = torch.cat([dst_all, nfeat], dim=-1)
        nfeat_new = self.node_mlp(cat_feat) + nfeat

        return efeat, nfeat_new


from typing import Union

import torch
import torch.nn as nn
# import transformer_engine as te
from dgl import DGLGraph
from torch import Tensor


class OneForecastProcessor(nn.Module):
  
    def __init__(
        self,
        aggregation: str = "sum",
        processor_layers: int = 16,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
        num_heads_edge: int = 4,
        num_heads_node: int = 4,
    ):
        super().__init__()

        layers = []
        for _ in range(processor_layers):
            edge_block = MeshEdgeBlockMultiHeadGated(
                input_dim_nodes=input_dim_nodes,
                input_dim_edges=input_dim_edges,
                output_dim=input_dim_edges,  
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                do_concat_trick=do_concat_trick,
                recompute_activation=recompute_activation,
                num_heads=num_heads_edge,
            )

            node_block = MeshNodeBlockMultiHeadAttn(
                aggregation=aggregation,
                input_dim_nodes=input_dim_nodes,
                input_dim_edges=input_dim_edges,
                output_dim=input_dim_nodes, 
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                recompute_activation=recompute_activation,
                num_heads=num_heads_node,
            )
            layers.append(edge_block)
            layers.append(node_block)

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)

        self.checkpoint_segments = [(0, self.num_processor_layers)]
        self.checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_segments(self, checkpoint_segments: int):
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError(
                    "Processor layers must be a multiple of checkpoint_segments"
                )
            seg_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, seg_size):
                self.checkpoint_segments.append((i, i + seg_size))
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_segments = [(0, self.num_processor_layers)]
            self.checkpoint_fn = set_checkpoint_fn(False)

    def run_segment(self, segment_layers):
        def segment_forward(efeat, nfeat, graph):
            for module in segment_layers:
                efeat, nfeat = module(efeat, nfeat, graph)
            return efeat, nfeat
        return segment_forward

    def forward(self, efeat: Tensor, nfeat: Tensor, graph: DGLGraph):
        for start, end in self.checkpoint_segments:
            efeat, nfeat = self.checkpoint_fn(
                self.run_segment(self.processor_layers[start:end]),
                efeat, nfeat, graph,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return efeat, nfeat


class OneForecastProcessorGraphTransformer(nn.Module):

    def __init__(
        self,
        attention_mask: torch.Tensor,
        num_attention_heads: int = 4,
        processor_layers: int = 16,
        input_dim_nodes: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        self.register_buffer("mask", self.attention_mask, persistent=False)

        layers = [
            te.pytorch.TransformerLayer(
                hidden_size=input_dim_nodes,
                ffn_hidden_size=hidden_dim,
                num_attention_heads=num_attention_heads,
                layer_number=i + 1,
                fuse_qkv_params=False,
            )
            for i in range(processor_layers)
        ]
        self.processor_layers = nn.ModuleList(layers)

    def forward(
        self,
        nfeat: Tensor,
    ) -> Tensor:
        nfeat = nfeat.unsqueeze(1)
        # TODO make sure reshaping the last dim to (h, d) is done automatically in the transformer layer
        for module in self.processor_layers:
            nfeat = module(
                nfeat,
                attention_mask=self.mask,
                self_attn_mask_type="arbitrary",
            )

        return torch.squeeze(nfeat, 1)



import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

try:
    from typing import Self
except ImportError:
    # for Python versions < 3.11
    from typing_extensions import Self



logger = logging.getLogger(__name__)


def get_lat_lon_partition_separators(partition_size: int):
  

    def _divide(num_lat_chunks: int, num_lon_chunks: int):
        if (num_lon_chunks * num_lat_chunks) != partition_size:
            raise ValueError(
                "Can't divide lat-lon grid into grid {num_lat_chunks} x {num_lon_chunks} chunks for partition_size={partition_size}."
            )
       
        lat_bin_width = 180.0 / num_lat_chunks
        lon_bin_width = 360.0 / num_lon_chunks

        lat_ranges = []
        lon_ranges = []

        for p_lat in range(num_lat_chunks):
            for p_lon in range(num_lon_chunks):
                lat_ranges += [
                    (lat_bin_width * p_lat - 90.0, lat_bin_width * (p_lat + 1) - 90.0)
                ]
                lon_ranges += [
                    (lon_bin_width * p_lon - 180.0, lon_bin_width * (p_lon + 1) - 180.0)
                ]

        lat_ranges[-1] = (lat_ranges[-1][0], None)
        lon_ranges[-1] = (lon_ranges[-1][0], None)

        return lat_ranges, lon_ranges

    lat_chunks, lon_chunks, i = 1, partition_size, 0
    while lat_chunks < lon_chunks:
        i += 1
        if partition_size % i == 0:
            lat_chunks = i
            lon_chunks = partition_size // lat_chunks

    lat_ranges, lon_ranges = _divide(lat_chunks, lon_chunks)

    if (lat_ranges is None) or (lon_ranges is None):
        raise ValueError("unexpected error, abort")

    min_seps = []
    max_seps = []

    for i in range(partition_size):
        lat = lat_ranges[i]
        lon = lon_ranges[i]
        min_seps.append([lat[0], lon[0]])
        max_seps.append([lat[1], lon[1]])

    return min_seps, max_seps


@dataclass
class MetaData(ModelMetaData):
    name: str = "OneForecast"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class OneForecast(Module):

    def __init__(
        self,
        params,
        mesh_level: Optional[int] = 5,
        multimesh_level: Optional[int] = None,
        multimesh: bool = True,
        input_res: tuple = (120, 240),
        input_dim_grid_nodes: int = 69,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 69,
        processor_type: str = "MessagePassing",
        khop_neighbors: int = 32,
        num_attention_heads: int = 4,
        processor_layers: int = 16,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        activation_fn: str = "silu",
        norm_type: str = "LayerNorm",
        use_cugraphops_encoder: bool = False,
        use_cugraphops_processor: bool = False,
        use_cugraphops_decoder: bool = False,
        do_concat_trick: bool = True,
        recompute_activation: bool = True,
        partition_size: int = 1,
        partition_group_name: Optional[str] = None,
        use_lat_lon_partitioning: bool = False,
        expect_partitioned_input: bool = False,
        global_features_on_rank_0: bool = False,
        produce_aggregated_output: bool = True,
        produce_aggregated_output_on_all_ranks: bool = True,
    ):
        super().__init__(meta=MetaData())

        # 'multimesh_level' deprecation handling
        if multimesh_level is not None:
            warnings.warn(
                "'multimesh_level' is deprecated and will be removed in a future version. Use 'mesh_level' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mesh_level = multimesh_level

        self.processor_type = processor_type
        if self.processor_type == "MessagePassing":
            khop_neighbors = 0
        self.is_distributed = False
        if partition_size > 1:
            self.is_distributed = True
        self.expect_partitioned_input = expect_partitioned_input
        self.global_features_on_rank_0 = global_features_on_rank_0
        self.produce_aggregated_output = produce_aggregated_output
        self.produce_aggregated_output_on_all_ranks = (
            produce_aggregated_output_on_all_ranks
        )
        self.partition_group_name = partition_group_name

        # create the lat_lon_grid
        self.latitudes = torch.linspace(-90, 90, steps=input_res[0] + 1)[:-1]
        self.longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]

        self.lat_lon_grid = torch.stack(
            torch.meshgrid(self.latitudes, self.longitudes, indexing="ij"), dim=-1
        )

        # Set activation function
        activation_fn = get_activation(activation_fn)

        # construct the graph
        self.graph = Graph(self.lat_lon_grid, mesh_level, multimesh, khop_neighbors, do_local_refine=True, refine_specs=[
        (0.0, 30.0, 105.0, 160.0, 1),(10.0, 30.0, -95.0, -35.0, 1),], complex_edge = False, lon_lat_complex_edge=[(23.5, -88.7, 25.1, -80.2),(37.0, -75.0, 33.0, -73.0),
    ])

        self.mesh_graph, self.attn_mask = self.graph.create_mesh_graph(verbose=False)
        self.g2m_graph = self.graph.create_g2m_graph(verbose=False)
        self.m2g_graph = self.graph.create_m2g_graph(verbose=False)

        self.g2m_edata = self.g2m_graph.edata["x"]
        self.m2g_edata = self.m2g_graph.edata["x"]
        self.mesh_ndata = self.mesh_graph.ndata["x"]
        if self.processor_type == "MessagePassing":
            self.mesh_edata = self.mesh_graph.edata["x"]
        elif self.processor_type == "GraphTransformer":
            # Dummy tensor to avoid breaking the API
            self.mesh_edata = torch.zeros((1, input_dim_edges))
        else:
            raise ValueError(f"Invalid processor type {processor_type}")

        if use_cugraphops_encoder or self.is_distributed:
            kwargs = {}
            if use_lat_lon_partitioning:
                min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
                kwargs = {
                    "src_coordinates": self.g2m_graph.srcdata["lat_lon"],
                    "dst_coordinates": self.g2m_graph.dstdata["lat_lon"],
                    "coordinate_separators_min": min_seps,
                    "coordinate_separators_max": max_seps,
                }
            self.g2m_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.g2m_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
                partition_by_bbox=use_lat_lon_partitioning,
                **kwargs,
            )
            self.g2m_edata = self.g2m_edata[edge_perm]

            if self.is_distributed:
                self.g2m_edata = self.g2m_graph.get_edge_features_in_partition(
                    self.g2m_edata
                )

        if use_cugraphops_decoder or self.is_distributed:
            kwargs = {}
            if use_lat_lon_partitioning:
                min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
                kwargs = {
                    "src_coordinates": self.m2g_graph.srcdata["lat_lon"],
                    "dst_coordinates": self.m2g_graph.dstdata["lat_lon"],
                    "coordinate_separators_min": min_seps,
                    "coordinate_separators_max": max_seps,
                }

            self.m2g_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.m2g_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
                partition_by_bbox=use_lat_lon_partitioning,
                **kwargs,
            )
            self.m2g_edata = self.m2g_edata[edge_perm]

            if self.is_distributed:
                self.m2g_edata = self.m2g_graph.get_edge_features_in_partition(
                    self.m2g_edata
                )

        if use_cugraphops_processor or self.is_distributed:
            kwargs = {}
            if use_lat_lon_partitioning:
                min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
                kwargs = {
                    "src_coordinates": self.mesh_graph.ndata["lat_lon"],
                    "dst_coordinates": self.mesh_graph.ndata["lat_lon"],
                    "coordinate_separators_min": min_seps,
                    "coordinate_separators_max": max_seps,
                }

            self.mesh_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.mesh_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
                partition_by_bbox=use_lat_lon_partitioning,
                **kwargs,
            )
            self.mesh_edata = self.mesh_edata[edge_perm]
            if self.is_distributed:
                self.mesh_edata = self.mesh_graph.get_edge_features_in_partition(
                    self.mesh_edata
                )
                self.mesh_ndata = self.mesh_graph.get_dst_node_features_in_partition(
                    self.mesh_ndata
                )

        self.input_dim_grid_nodes = input_dim_grid_nodes
        self.output_dim_grid_nodes = output_dim_grid_nodes
        self.input_res = input_res

        # by default: don't checkpoint at all
        self.model_checkpoint_fn = set_checkpoint_fn(False)
        self.encoder_checkpoint_fn = set_checkpoint_fn(False)
        self.decoder_checkpoint_fn = set_checkpoint_fn(False)

        # initial feature embedder
        self.encoder_embedder = OneForecastEncoderEmbedder(
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )
        self.decoder_embedder = OneForecastDecoderEmbedder(
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # grid2mesh encoder
        self.encoder = MeshGraphEncoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_src_nodes=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # icosahedron processor
        if processor_layers <= 2:
            raise ValueError("Expected at least 3 processor layers")
        if processor_type == "MessagePassing":
            self.processor_encoder = OneForecastProcessor(
                aggregation=aggregation,
                processor_layers=1,
                input_dim_nodes=hidden_dim,
                input_dim_edges=hidden_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                do_concat_trick=do_concat_trick,
                recompute_activation=recompute_activation,
            )
            self.processor = OneForecastProcessor(
                aggregation=aggregation,
                processor_layers=processor_layers - 2,
                input_dim_nodes=hidden_dim,
                input_dim_edges=hidden_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                do_concat_trick=do_concat_trick,
                recompute_activation=recompute_activation,
            )
            self.processor_decoder = OneForecastProcessor(
                aggregation=aggregation,
                processor_layers=1,
                input_dim_nodes=hidden_dim,
                input_dim_edges=hidden_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                do_concat_trick=do_concat_trick,
                recompute_activation=recompute_activation,
            )
        else:
            self.processor_encoder = torch.nn.Identity()
            self.processor = OneForecastProcessorGraphTransformer(
                attention_mask=self.attn_mask,
                num_attention_heads=num_attention_heads,
                processor_layers=processor_layers,
                input_dim_nodes=hidden_dim,
                hidden_dim=hidden_dim,
            )
            self.processor_decoder = torch.nn.Identity()

        # mesh2grid decoder
        self.decoder = MeshGraphDecoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # final MLP
        self.finale = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=output_dim_grid_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

    def set_checkpoint_model(self, checkpoint_flag: bool):
       
        self.model_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)
        if checkpoint_flag:
            self.processor.set_checkpoint_segments(-1)
            self.encoder_checkpoint_fn = set_checkpoint_fn(False)
            self.decoder_checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_processor(self, checkpoint_segments: int):
       
        self.processor.set_checkpoint_segments(checkpoint_segments)

    def set_checkpoint_encoder(self, checkpoint_flag: bool):
       
        self.encoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def set_checkpoint_decoder(self, checkpoint_flag: bool):
       
        self.decoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def encoder_forward(
        self,
        grid_nfeat: Tensor,
    ) -> Tensor:
       
        # embedd graph features
        (
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            g2m_efeat_embedded,
            mesh_efeat_embedded,
        ) = self.encoder_embedder(
            grid_nfeat,
            self.mesh_ndata,
            self.g2m_edata,
            self.mesh_edata,
        )

        # encode lat/lon to multimesh
        grid_nfeat_encoded, mesh_nfeat_encoded = self.encoder(
            g2m_efeat_embedded,
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            self.g2m_graph,
        )

        # process multimesh graph
        if self.processor_type == "MessagePassing":
            mesh_efeat_processed, mesh_nfeat_processed = self.processor_encoder(
                mesh_efeat_embedded,
                mesh_nfeat_encoded,
                self.mesh_graph,
            )
        else:
            mesh_nfeat_processed = self.processor_encoder(
                mesh_nfeat_encoded,
            )
            mesh_efeat_processed = None
        return mesh_efeat_processed, mesh_nfeat_processed, grid_nfeat_encoded

    def decoder_forward(
        self,
        mesh_efeat_processed: Tensor,
        mesh_nfeat_processed: Tensor,
        grid_nfeat_encoded: Tensor,
    ) -> Tensor:
       

        # process multimesh graph
        if self.processor_type == "MessagePassing":
            _, mesh_nfeat_processed = self.processor_decoder(
                mesh_efeat_processed,
                mesh_nfeat_processed,
                self.mesh_graph,
            )
        else:
            mesh_nfeat_processed = self.processor_decoder(
                mesh_nfeat_processed,
            )

        m2g_efeat_embedded = self.decoder_embedder(self.m2g_edata)

        # decode multimesh to lat/lon
        grid_nfeat_decoded = self.decoder(
            m2g_efeat_embedded, grid_nfeat_encoded, mesh_nfeat_processed, self.m2g_graph
        )

        # map to the target output dimension
        grid_nfeat_finale = self.finale(
            grid_nfeat_decoded,
        )

        return grid_nfeat_finale

    def custom_forward(self, grid_nfeat: Tensor) -> Tensor:
       
        (
            mesh_efeat_processed,
            mesh_nfeat_processed,
            grid_nfeat_encoded,
        ) = self.encoder_checkpoint_fn(
            self.encoder_forward,
            grid_nfeat,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        # checkpoint of processor done in processor itself
        if self.processor_type == "MessagePassing":
            mesh_efeat_processed, mesh_nfeat_processed = self.processor(
                mesh_efeat_processed,
                mesh_nfeat_processed,
                self.mesh_graph,
            )
        else:
            mesh_nfeat_processed = self.processor(
                mesh_nfeat_processed,
            )
            mesh_efeat_processed = None

        grid_nfeat_finale = self.decoder_checkpoint_fn(
            self.decoder_forward,
            mesh_efeat_processed,
            mesh_nfeat_processed,
            grid_nfeat_encoded,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        return grid_nfeat_finale

    def forward(
        self,
        grid_nfeat: Tensor,
    ) -> Tensor:
        invar = self.prepare_input(
            grid_nfeat, self.expect_partitioned_input, self.global_features_on_rank_0
        )
        outvar = self.model_checkpoint_fn(
            self.custom_forward,
            invar,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        outvar = self.prepare_output(
            outvar,
            self.produce_aggregated_output,
            self.produce_aggregated_output_on_all_ranks,
        )
        return outvar

    def prepare_input(
        self,
        invar: Tensor,
        expect_partitioned_input: bool,
        global_features_on_rank_0: bool,
    ) -> Tensor:
      
        if global_features_on_rank_0 and expect_partitioned_input:
            raise ValueError(
                "global_features_on_rank_0 and expect_partitioned_input cannot be set at the same time."
            )

        if not self.is_distributed:
            if invar.size(0) != 1:
                raise ValueError("OneForecast does not support batch size > 1")
            invar = invar[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)

        else:
            # is_distributed
            if not expect_partitioned_input:
                # global_features_on_rank_0
                if invar.size(0) != 1:
                    raise ValueError("OneForecast does not support batch size > 1")

                invar = invar[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)

                # scatter global features
                invar = self.g2m_graph.get_src_node_features_in_partition(
                    invar,
                    scatter_features=global_features_on_rank_0,
                )

        return invar

    def prepare_output(
        self,
        outvar: Tensor,
        produce_aggregated_output: bool,
        produce_aggregated_output_on_all_ranks: bool = True,
    ) -> Tensor:
       
        if produce_aggregated_output or not self.is_distributed:
            # default case: output of shape [N, C, H, W]
            if self.is_distributed:
                outvar = self.m2g_graph.get_global_dst_node_features(
                    outvar,
                    get_on_all_ranks=produce_aggregated_output_on_all_ranks,
                )

            outvar = outvar.permute(1, 0)
            outvar = outvar.view(self.output_dim_grid_nodes, *self.input_res)
            outvar = torch.unsqueeze(outvar, dim=0)

        return outvar

    def to(self, *args: Any, **kwargs: Any) -> Self:
        
        self = super(OneForecast, self).to(*args, **kwargs)

        self.g2m_edata = self.g2m_edata.to(*args, **kwargs)
        self.m2g_edata = self.m2g_edata.to(*args, **kwargs)
        self.mesh_ndata = self.mesh_ndata.to(*args, **kwargs)
        self.mesh_edata = self.mesh_edata.to(*args, **kwargs)

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.g2m_graph = self.g2m_graph.to(device)
        self.mesh_graph = self.mesh_graph.to(device)
        self.m2g_graph = self.m2g_graph.to(device)

        return self

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = OneForecast().to(device)

    input = torch.randn(1, 69, 120, 240).to(device)
    output = net(input)

    print(output.shape)
