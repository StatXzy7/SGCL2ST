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

