import torch
from torch import nn
from torch_geometric.nn import TransformerConv, GCNConv, Sequential, BatchNorm, LayerNorm, InstanceNorm
from torch_geometric.utils import dense_to_sparse
from transformer import *
import timm
import config as CFG

import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable,
        n_genes = 785, encode_dim = 1024,
        channel=32, patch_size=7, kernel_size=5, conv_layers = 2,
        heads=8, dim_head=64, mlp_dim=1024, dropout = 0.2, depth_trans = 8, depth_gnn = 2
    ):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name, pretrained, num_classes=0, global_pool="avg"
        # )
        # for p in self.model.parameters():
        #     p.requires_grad = trainable
        
        self.patch_embedding = nn.Conv2d(3,channel,patch_size,patch_size) # 112/7 = 16
        self.conv2d_layers = nn.Sequential(*[Conv2d_block(channel,kernel_size) for i in range(conv_layers)],) 
        self.flat=nn.Sequential(
            nn.Conv2d(channel,channel//8,1,1), # 4*16*16
            nn.Flatten(),
        )
        # self.gnn = Sequential('x, edge_index', [
        #     (TransformerConv(encode_dim, encode_dim), 'x, edge_index -> x1'),
        #     BatchNorm(encode_dim),
        #     nn.ReLU(), 
        #     ])
        self.gnn_layers = nn.ModuleList()
        for i in range(depth_gnn):       
            gnn = Sequential('x, edge_index', [
                        (TransformerConv(encode_dim, encode_dim), 'x, edge_index -> x1'),
                        BatchNorm(encode_dim),
                        nn.ReLU(), 
                        ])
            self.gnn_layers.append(gnn)
        
        self.Transformer = nn.Sequential(*[attn_block(encode_dim,heads,dim_head,mlp_dim,dropout) for i in range(depth_trans)])
        
    def forward(self, x, adj):
        x = self.patch_embedding(x)
        x = self.conv2d_layers(x)
        x = self.flat(x)
        pos = x
        for layer in self.gnn_layers:
            pos = layer(pos,adj)
        x = self.Transformer((x + pos).unsqueeze(0)).squeeze(0)
        return x

class SpotEncoder(nn.Module):
    """
    Encode spots gene expression to a fixed size vector using GCN layers.
    """
    
    class GCNLayer(nn.Module):
        def __init__(self, input_dim = CFG.spot_embedding, hidden_dim = CFG.spot_embedding):
            super(SpotEncoder.GCNLayer, self).__init__()
            self.gcn = GCNConv(input_dim, hidden_dim)
            self.layer_norm = nn.LayerNorm(input_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x, edge_index):
            x = self.layer_norm(x)
            x = self.relu(x)
            x = self.gcn(x, edge_index)
            return x
    
    class ResGCNLayer(nn.Module):
        def __init__(self, dim):
            super(SpotEncoder.ResGCNLayer, self).__init__()
            self.layer1 = SpotEncoder.GCNLayer(dim, dim)
            self.layer2 = SpotEncoder.GCNLayer(dim, dim)

        def forward(self, x, edge_index):
            x = self.layer1(x, edge_index)
            x = self.layer2(x, edge_index)
            return x
    
    def __init__(self, input_dim = CFG.spot_embedding, hidden_dim = CFG.spot_embedding, num_layers=3):
        super(SpotEncoder, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # Input GCN layer
        self.convs.append(SpotEncoder.GCNLayer(input_dim, hidden_dim))
        
        # Hidden ResGCN layers
        for _ in range((num_layers-1)//2):
            self.convs.append(SpotEncoder.ResGCNLayer(hidden_dim))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index) + x
        return x

class SpotAutoEncoder(nn.Module):
    ## https://github.com/JiangBioLab/DeepST
    # Input: Gene Expression N * n_genes
    def __init__(self, 
                GNN_type = 'TransformerConv',
                n_genes = CFG.spot_embedding, encode_dim = 256, projection_dim = CFG.projection_dim,
                channel=32, patch_size=7, kernel_size=5, conv_layers = 2,
                heads=8, dim_head=64, mlp_dim=256, dropout = 0.2, depth_trans = 8,
                ):
        super(SpotAutoEncoder, self).__init__()
        
        self.GNN_type = GNN_type
        self.linear_dim = [n_genes, encode_dim]
        self.gnn_dim = [n_genes, encode_dim]

        ######### Encoders        
        ### Linear embedding
        # self.linear = nn.Sequential(
        #     nn.Linear(encode_dim, encode_dim),
        #     nn.LayerNorm(encode_dim),
        #     nn.ReLU(), 
        # )
        ### GNN position embedding
        self.gnn = Sequential('x, edge_index', [
                            (TransformerConv(encode_dim, encode_dim), 'x, edge_index -> x1'),
                            LayerNorm(encode_dim),
                            nn.ReLU(), 
                            ])
        
        #### Transformer Encoders
        self.Transformer = nn.Sequential(*[attn_block(encode_dim,heads,dim_head,mlp_dim,dropout) for i in range(depth_trans)])
        
        ######### Decoders
        ### Gene Decoders
        self.gene_head = nn.Sequential(
            nn.LayerNorm(encode_dim),
            nn.Linear(encode_dim, n_genes),
        )
        ### ZINB Decoders
        self.mean = nn.Sequential(nn.Linear(encode_dim, n_genes), MeanAct())
        self.disp = nn.Sequential(nn.Linear(encode_dim, n_genes), DispAct())
        self.pi = nn.Sequential(nn.Linear(encode_dim, n_genes), nn.Sigmoid())


    def encode(
        self, x, adj,
        ):
        # print("x.shape",x.shape)
        # print("adj.shape",adj.shape)
        gnn_x = self.gnn(x, adj)
        trans_x = self.Transformer((x + gnn_x).unsqueeze(0)).squeeze(0)

        return trans_x

    def decode(
        self, x,
        ):
        # print("x.shape",x.shape)
        result = self.gene_head(x)
        m = self.mean(x)
        d = self.disp(x)
        p = self.pi(x)
        extra=(m,d,p)
        
        return result, extra
    
############################################

class ImageEncoder_resnet18(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet18', pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
    
class ImageEncoder_resnet50(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim, 
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

##### NB_module

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
    
def NB_loss(x, h_r, h_p):
        
        ll = torch.lgamma(torch.exp(h_r) + x) - torch.lgamma(torch.exp(h_r))
        ll += h_p * x - torch.log(torch.exp(h_p) + 1) * (x + torch.exp(h_r))

        loss = -torch.mean(torch.sum(ll, axis=-1))
        return loss

def ZINB_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
    eps = 1e-10
    if isinstance(scale_factor,float):
        scale_factor=np.full((len(mean),),scale_factor)
    scale_factor = scale_factor[:, None]
    mean = mean * scale_factor

    t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
    t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0-pi+eps)
    zero_nb = torch.pow(disp/(disp+mean+eps), disp)
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    if ridge_lambda > 0:
        ridge = ridge_lambda*torch.square(pi)
        result += ridge
    result = torch.mean(result)
    return result

class Conv2d_block(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        self.dw=nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
        )
        self.pw=nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
    def forward(self,x):
        x=self.dw(x)+x
        x=self.pw(x)
        return x
    
class Conv1d_block(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        self.rw=nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size, padding="same"),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size, padding="same"),
                nn.BatchNorm1d(dim),
                nn.GELU(),
        )
        self.pw=nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )
    def forward(self,x):
        x=self.rw(x)+x
        x=self.pw(x)
        return x