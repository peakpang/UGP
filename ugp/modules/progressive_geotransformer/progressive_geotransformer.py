import numpy as np
import torch
import torch.nn as nn

from ugp.modules.transformer.rpe_transformer import RPETransformerLayer
from ugp.modules.transformer.vanilla_transformer import TransformerLayer
from ugp.modules.transformer import SinusoidalPositionalEmbedding
from ugp.modules.ops import pairwise_distance

def _check_block_type(block):
    if block not in ['self', 'nearself','midself','farself']:
        raise ValueError('Unsupported block type "{}".'.format(block))

class Build_NMF(nn.Module):
    def __init__(self):
        super(Build_NMF, self).__init__()

    def forward(self, ref_points, src_points):
        r"""Extract superpoint correspondences.

        Args:
            ref_fps_feats (Tensor): NXC
            src_fps_feats (Tensor): NXC
            ref_feats_norm (Tensor): M1XC
            src_feats_norm (Tensor): M2XC
            
        Returns:
            ref_branch_src_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_branch_ref_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
           
        """
        
        ref_distances = torch.cdist(ref_points,ref_points)
        src_distances = torch.cdist(src_points,src_points)
        ref_values,ref_indices = torch.max(ref_distances,dim=-1)
        src_values,src_indices = torch.max(src_distances,dim=-1)
        
        ref_div_base = ref_values / 3
        src_div_base = src_values / 3
        
        ref_mask_list = []
        src_mask_list = []
        
        for i in range(2):
            if i == 0:
                ref_threshold = ref_div_base
                src_threshold = src_div_base
            else:
                ref_threshold = ref_div_base * 2
                src_threshold = src_div_base * 2
            # 生成掩码
            ref_masks = torch.ones(size=(ref_points.shape[0], ref_points.shape[0]), dtype=torch.bool).cuda()
            src_masks = torch.ones(size=(src_points.shape[0], src_points.shape[0]), dtype=torch.bool).cuda()

            ref_masks[ref_distances < ref_threshold.unsqueeze(1)] = False
            src_masks[src_distances < src_threshold.unsqueeze(1)] = False

            ref_mask_list.append(ref_masks)
            src_mask_list.append(src_masks)
  

        return ref_mask_list,src_mask_list

       
class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'cross':
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def forward(self, feats0, feats1,embeddings0, embeddings1,masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'nearself':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, attention_masks=masks0[0]) # 切记True表示不算\\要用attention_mask
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, attention_masks=masks1[0])
            elif block == 'midself':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, attention_masks=masks0[1])
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, attention_masks=masks1[1])
            elif block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, attention_masks=None)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, attention_masks=None)
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            attention_scores.append([scores0, scores1])
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1, attention_scores
        
class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')
        
    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)
        
        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings     
      
class ProgressiveGeoTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        input_idim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(ProgressiveGeoTransformer, self).__init__()
        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)
        self.in_proj = nn.Linear(input_dim, 128)
        self.in_proj_i = nn.Linear(input_idim, 64)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_ifeats,
        src_ifeats,
        ref_masks_list=None,
        src_masks_list=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        ref_ifeats = self.in_proj_i(ref_ifeats)
        src_ifeats = self.in_proj_i(src_ifeats)

        ref_fuse_feats = torch.cat([ref_feats,ref_ifeats],dim=-1)
        src_fuse_feats = torch.cat([src_feats,src_ifeats],dim=-1)

        ref_feats, src_feats,attention_scores= self.transformer(
            ref_fuse_feats,
            src_fuse_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks_list,
            masks1=src_masks_list,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, attention_scores