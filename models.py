import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import *


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

class CLIPModel_resnet50(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=2048,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet50()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, patches, centers, exps, adj, aug=False):
        image_features = self.image_encoder(patches)
        spot_features = exps
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class myModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding, #785,171
        projection_dim = CFG.projection_dim, #256 
    ):
        super().__init__()
        self.spot_embedding = spot_embedding
        # self.image_encoder = ImageEncoder()
        self.image_encoder = ImageEncoder_resnet50()
        self.spot_encoder = SpotEncoder()
        self.spot_autoencoder = SpotAutoEncoder(n_genes = spot_embedding)
        self.image_projection = ProjectionHead(embedding_dim = image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim = spot_embedding) #3467 shared hvgs  projection_dim spot_embedding
        self.temperature = temperature
        self.rmse = 0.1
        self.zinb = 0.25


    def forward(self, patches, centers, exps, adj, oris, sfs):
        # Getting Image and spot Features
        image_features = self.image_encoder(patches)
        # spot_features = exps
        spot_features = self.spot_encoder(exps, adj)
        # print("image_features.shape = ", image_features.shape)
        # print("spot_features.shape = ", spot_features.shape)
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)
        # print("image_embeddings.shape = ", image_embeddings.shape)
        # print("spot_embeddings.shape = ", spot_embeddings.shape)
        
        spot_encoding = self.spot_autoencoder.encode(spot_embeddings, adj)
        spot_reconstruction, extra = self.spot_autoencoder.decode(spot_encoding)
        # print("spot_encoding.shape = ", spot_encoding.shape)
        # print("spot_reconstruction.shape = ", spot_reconstruction.shape)
        
      
        # Calculating the Contrastive Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        contrastive_loss =  ((images_loss + spots_loss) / 2.0).mean() # shape: (batch_size)
        
        #### Calculating the Reconstruction Loss
        ## RMSE Loss
        rmse_loss = torch.sqrt(nn.MSELoss()(spot_reconstruction, exps))
        ## ZINB Loss
        m,d,p=extra
        zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
        ## Reconstruction_loss
        reconstruction_loss = self.rmse * rmse_loss + self.zinb * zinb_loss
        
        return contrastive_loss + reconstruction_loss, contrastive_loss, rmse_loss, zinb_loss

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
