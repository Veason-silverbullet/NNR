# download and extract MIND-large for MIND-200k
cd ..
mkdir MIND-200k
cd MIND-200k
mkdir download
cd download
wget -O MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
unzip MINDlarge_train.zip -d train
wget -O MINDlarge_dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip
unzip MINDlarge_dev.zip -d dev
wget -O wikidata-graph.zip https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip
unzip wikidata-graph.zip
cd ..
tree -L 2


# download and extract MIND-small
cd ..
mkdir MIND-small
cd MIND-small
mkdir download
cd download
wget -O MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
unzip MINDsmall_train.zip -d train
wget -O MINDsmall_dev.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_dev.zip -d dev
wget -O wikidata-graph.zip https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip
unzip wikidata-graph.zip
cd ..
tree -L 2


# download and extract MIND-large
cd ..
mkdir MIND-large
cd MIND-large
mkdir download
cd download
wget -O MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
unzip MINDlarge_train.zip -d train
wget -O MINDlarge_dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip
unzip MINDlarge_dev.zip -d dev
wget -O MINDlarge_test.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
unzip MINDlarge_test.zip -d test
wget -O wikidata-graph.zip https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip
unzip wikidata-graph.zip
cd ..
tree -L 2


# preprocess dataset
cd ../NNR
python prepare_MIND_dataset.py
