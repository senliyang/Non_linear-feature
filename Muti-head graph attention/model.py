import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import JumpingKnowledge
from layers import GraphAttentionLayer, InnerProductDecoder

class Graph(nn.Module):
    def __init__(self, n_in_features: int, n_hid_layers: int, hid_features: list, n_heads: list, n_drug: int,
                 n_cir: int,add_layer_attn: bool, residual: bool, dropout: float = 0.6):
        super(Graph, self).__init__()
        assert n_hid_layers == len(hid_features) == len(n_heads), f'Enter valid arch params.'
        self.n_drug = n_drug
        self.n_cir = n_cir
        self.n_hid_layers = n_hid_layers
        self.hid_features = hid_features
        self.dropout = nn.Dropout(dropout)
        self.att = nn.Parameter(torch.rand(self.n_hid_layers + 1), requires_grad=True)
        self.att_cir = nn.Parameter(torch.rand(self.n_hid_layers + 1), requires_grad=True)
        self.att2 = nn.Parameter(torch.rand(2), requires_grad=True)
        self.reconstructions = InnerProductDecoder(
            name='gan_decoder',
            input_dim=hid_features[0], num_c=self.n_cir, act=torch.sigmoid)

        self.CNN_hetero = nn.Conv1d(in_channels=self.n_hid_layers + 1,
                                    out_channels=hid_features[0],
                                    kernel_size=(hid_features[0], 1),
                                    stride=1,
                                    bias=True)
        self.CNN_cir = nn.Conv1d(in_channels=self.n_hid_layers + 1,
                                 out_channels=hid_features[0],
                                 kernel_size=(hid_features[0], 1),
                                 stride=1,
                                 bias=True)
        # stack graph attention layers
        self.conv = nn.ModuleList()
        self.conv_cir = nn.ModuleList()
        tmp = [n_in_features] + hid_features
        tmp_cir = [n_cir] + hid_features

        for i in range(n_hid_layers):
            self.conv.append(
                GraphAttentionLayer(tmp[i], tmp[i + 1], n_heads[i], residual=residual),
            )
            self.conv_cir.append(
                GraphAttentionLayer(tmp_cir[i], tmp_cir[i + 1], n_heads[i], residual=residual),
            )

        if n_in_features != hid_features[0]:
            self.proj = Linear(n_in_features, hid_features[0], weight_initializer='glorot', bias=True)
            self.proj_cir = Linear(n_cir, hid_features[0], weight_initializer='glorot', bias=True)

        else:
            self.register_parameter('proj', None)

        if add_layer_attn:
            self.JK = JumpingKnowledge('lstm', tmp[-1], n_hid_layers + 1)
            self.JK_cir = JumpingKnowledge('lstm', tmp_cir[-1], n_hid_layers + 1)
        else:
            self.register_parameter('JK', None)

        if self.proj is not None:
            self.proj.reset_parameters()

    def forward(self, x, edge_idx,  x_cir, edge_idx_cir):
        # encoder
        embd_tmp = x
        cnn_embd_hetro = self.proj(embd_tmp) if self.proj is not None else embd_tmp
        for i in range(self.n_hid_layers):
            embd_tmp = self.conv[i](embd_tmp, edge_idx)
            cnn_embd_hetro = torch.cat((cnn_embd_hetro, embd_tmp), 1)
        cnn_embd_hetro = cnn_embd_hetro.t().view(1, self.n_hid_layers + 1, self.hid_features[0],
                                                 self.n_drug + self.n_cir)
        cnn_embd_hetro = self.CNN_hetero(cnn_embd_hetro)
        cnn_embd_hetro = cnn_embd_hetro.view(self.hid_features[0], self.n_drug + self.n_cir).t()

        embd_tmp_cir = x_cir
        cnn_embd_cir = self.proj_cir(embd_tmp_cir) if self.proj_cir is not None else embd_tmp_cir
        for i in range(self.n_hid_layers):
            embd_tmp_cir = self.conv_cir[i](embd_tmp_cir, edge_idx_cir)
            cnn_embd_cir = torch.cat((cnn_embd_cir, embd_tmp_cir), 1)
        cnn_embd_cir = cnn_embd_cir.t().view(1, self.n_hid_layers + 1, self.hid_features[0], self.n_cir)
        cnn_embd_cir = self.CNN_cir(cnn_embd_cir)
        cnn_embd_cir = cnn_embd_cir.view(self.hid_features[0], self.n_cir).t()

        embd_heter = cnn_embd_hetro
        embd_cir = cnn_embd_cir
        final_embd = self.dropout(embd_heter)
        embd_cir = self.dropout(embd_cir)

        ret, ass_c = self.reconstructions(final_embd, embd_cir)
        return ret.reshape(218, 271), ass_c
