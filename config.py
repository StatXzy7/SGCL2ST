

#commented out variables are handled by argparse in main.py
debug = True
# batch_size = 128
# num_workers = 0
lr = 1e-3 #
weight_decay = 1e-3 #
patience = 5
step = 30
factor = 0.2

# HER2  patience = 5 factor = 0.2
# epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'auto'
image_embedding = 2048
spot_embedding = 171 #HER2:785, cSCC:171

pretrained = False
trainable = True 
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
