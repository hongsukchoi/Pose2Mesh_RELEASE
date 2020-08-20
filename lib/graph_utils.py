# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Graph utilities
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
from coarsening import coarsen, laplacian, perm_index_reverse, lmax_L, rescale_L


def normalize_sparse_mx(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_graph(hand_tri, num_vertex):
    """
    :param hand_tri: T x 3
    :return: adj: sparse matrix, V x V (torch.sparse.FloatTensor)
    """
    num_tri = hand_tri.shape[0]
    edges = np.empty((num_tri * 3, 2))
    for i_tri in range(num_tri):
        edges[i_tri * 3] = hand_tri[i_tri, :2]
        edges[i_tri * 3 + 1] = hand_tri[i_tri, 1:]
        edges[i_tri * 3 + 2] = hand_tri[i_tri, [0, 2]]

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_vertex, num_vertex), dtype=np.float32)

    adj = adj - (adj > 1) * 1.0

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # adj = normalize_sparse_mx(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def build_adj(joint_num, skeleton, flip_pairs):
    adj_matrix = np.zeros((joint_num, joint_num))
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1
    for lr in flip_pairs:
        adj_matrix[lr] = 1
        adj_matrix[lr[1], lr[0]] = 1

    return adj_matrix + np.eye(joint_num)


def build_coarse_graphs(mesh_face, joint_num, skeleton, flip_pairs, levels=9):
    joint_adj = build_adj(joint_num, skeleton, flip_pairs)
    # Build graph
    mesh_adj = build_graph(mesh_face, mesh_face.max() + 1)
    graph_Adj, graph_L, graph_perm = coarsen(mesh_adj, levels=levels)
    input_Adj = sp.csr_matrix(joint_adj)
    input_Adj.eliminate_zeros()
    input_L = laplacian(input_Adj, normalized=True)

    graph_L[-1] = input_L
    graph_Adj[-1] = input_Adj

    # Compute max eigenvalue of graph Laplacians, rescale Laplacian
    graph_lmax = []
    renewed_lmax = []
    for i in range(levels):
        graph_lmax.append(lmax_L(graph_L[i]))
        graph_L[i] = rescale_L(graph_L[i], graph_lmax[i])
    #     renewed_lmax.append(lmax_L(graph_L[i]))

    return graph_Adj, graph_L, graph_perm, perm_index_reverse(graph_perm[0])


def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))

    return L


class my_sparse_mm(torch.autograd.Function):
    """
    this function is forked from https://github.com/xbresson/spectral_graph_convnets
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        print("CHECK sparse W: ", W.is_cuda)
        print("CHECK sparse x: ", x.is_cuda)
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx
