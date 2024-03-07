cd data
# Get Her2ST dataset
git clone https://github.com/almaan/her2st.git
# Get cSCC dataset
wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE144240&format=file" -O GSE144240_RAW.tar


# unzip the data used
gzip -d ./her2st/data/ST-cnts/*.tsv.gz

mkdir -p ./GSE144240_RAW  
tar -xvf ./GSE144240_RAW.tar -C GSE144240_RAW/ 
cd GSE144240_RAW  
for file in *.gz; do gunzip -f "$file"; done

# Get SGCL2ST model
# https://onedrive.live.com/?id=6F7A588E449ED6AC%21962&cid=6F7A588E449ED6AC
cd model
wget "https://public.sn.files.1drv.com/y4mKKprWN7l5sTKkLF3lwV3Tj1KoCWbx2VUw0cnic-0GXWzLmlVgzMBNExT7LB8490JmLRtY5sAu1B9K_LgIrU1Ft_kvbbCfwmV_9_K0DMoFaItx9f8beZIHUv6UQmHBV9Qp6q3_gWn9--8vjCj5qVgxjc7tjcdLMmg1aHQTf_VzqHmfAtREtITVHjr7aVKZ409-jSO4oxT1IuKjkkTUbuxY7t8rIRjOs5BKDsMZBi5X7A?AVOverride=1" -O ICL2ST_cSCC.pt
wget "https://public.sn.files.1drv.com/y4mY2KthEUiV8HxqupGcVTA8fXK0fvbk3ofmAh-ayiS-9CheV5RDALka-9RMb_rVth0b7_K7_y8OFlEwXGKuNrtM0MvOQaxt17rAga2RPPa5ZxW7IXHW8WSZy0pig9oyNvJbpu89rUeLSFxE99bdHRbMQNe4UHPle3ZXkdAoY2lLhF2O25O4nCENL9pBHMX1z0ggAIN89Oh4tA4sHeVk6acJnf4bfSnuB_g49-fU7PZFKk?AVOverride=1" -O ICL2ST_HER2.pt
