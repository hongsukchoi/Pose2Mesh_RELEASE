# codes changed from https://github.com/xbresson/spectral_graph_convnets
import torch


def graph_conv_cheby(x, cl, bn, L, Fout, K):
    # parameters
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = Chebyshev order & support size
    B, V, Fin = x.size()
    B, V, Fin = int(B), int(V), int(Fin)

    # transform to Chebyshev basis
    x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin * B])  # V x Fin*B
    x = x0.unsqueeze(0)  # 1 x V x Fin*B

    def concat(x, x_):
        x_ = x_.unsqueeze(0)  # 1 x V x Fin*B
        return torch.cat((x, x_), 0)  # K x V x Fin*B

    if K > 1:
        x1 = torch.sparse.mm(L, x0)  # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for k in range(2, K):
        x2 = 2 * torch.sparse.mm(L, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2

    x = x.view([K, V, Fin, B])  # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B * V, Fin * K])  # B*V x Fin*K

    # Compose linearly Fin features to get Fout features
    x = cl(x)  # B*V x Fout
    if bn is not None:
        x = bn(x)  # B*V x Fout
    x = x.view([B, V, Fout])  # B x V x Fout

    return x
