# Neural News Recommendation
This repository is for the paper [**Neural News Recommendation with Collaborative News Encoding and Structural User Encoding** (EMNLP-2021 Finding)](https://arxiv.org/pdf/2109.00750.pdf).


## Dataset Preparation
The experiments are conducted on the 200K-MIND dataset. Our code will try to download and sample the 200K-MIND dataset to the directory `../MIND` (see Line 119 of `config.py` and `download_sample_MIND.py`).

Since the MIND dataset is quite large, if our code cannot download it successfully due to unstable network connection, please execute the shell file `download_sample_MIND.sh` instead. If the automatic download still fails, we recommend to download the MIND dataset and knowledge graph manually according to the links in `download_sample_MIND.sh`.

Assume that now the pwd is `./NNR`, the downloaded and extracted MIND dataset should be organized as

    (terminal) $ bash download_sample_MIND.sh # Assume this command is executed successfully
    (terminal) $ cd ../MIND
    (terminal) $ tree -L 2
    (terminal) $ .
                 ├── dev
                 │   ├── behaviors.tsv
                 │   ├── entity_embedding.vec
                 │   ├── news.tsv
                 │   ├── __placeholder__
                 │   └── relation_embedding.vec
                 ├── dev.zip
                 ├── train
                 │   ├── behaviors.tsv
                 │   ├── entity_embedding.vec
                 │   ├── news.tsv
                 │   ├── __placeholder__
                 │   └── relation_embedding.vec
                 ├── train.zip
                 ├── wikidata-graph
                 │   ├── description.txt
                 │   ├── label.txt
                 │   └── wikidata-graph.tsv
                 └── wikidata-graph.zip


## Environment Requirements
    (terminal) $ pip install -r requirements.txt

Our experiments require python3 and torch>=1.9.0. The [torch_scatter](https://github.com/rusty1s/pytorch_scatter) package is also neccessary. Our code will try to install it automatically (see Line 11 of `userEncoders.py`).
If the automatic installation fails, please follow [https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) to install the package manually.


## Experiment Running
<hr>Our Model
<pre><code>python main.py --news_encoder=CNE --user_encoder=SUE</code></pre>

<hr>Neural news recommendation baselines in Section 4.2
<pre><code>python main.py --news_encoder=DAE       --user_encoder=GRU
python main.py --news_encoder=Inception --user_encoder=CATT  --category_embedding_dim=300 --subCategory_embedding_dim=300
python main.py --news_encoder=KCNN      --user_encoder=CATT  --word_embedding_dim=100 --entity_embedding_dim=100 --context_embedding_dim=100
python main.py --news_encoder=CNN       --user_encoder=LSTUR
python main.py --news_encoder=NAML      --user_encoder=ATT
python main.py --news_encoder=PNE       --user_encoder=PUE
python main.py --news_encoder=MHSA      --user_encoder=MHSA
python main.py --news_encoder=HDC       --user_encoder=FIM   --click_predictor=FIM</code></pre>

<hr>General news recommendation baselines in Section 4.2
<pre><code>cd general_recommendation_methods
python generate_tf_idf_feature_file.py
python generate_libfm_data.py
python libfm_main.py
python DSSM_main.py 
python wide_deep_main.py</code></pre>


<hr>Variants of our model in Section 4.2
<pre><code>python main.py --news_encoder=CNE_wo_CS --user_encoder=SUE
python main.py --news_encoder=CNE_wo_CA --user_encoder=SUE
python main.py --news_encoder=CNE       --user_encoder=SUE_wo_GCN
python main.py --news_encoder=CNE       --user_encoder=SUE_wo_HCA</code></pre>


<hr>Ablation experiments for news encoding in Section 5.2
<pre><code>python main.py --news_encoder=CNN          --user_encoder=ATT
python main.py --news_encoder=KCNN         --user_encoder=ATT --word_embedding_dim=100 --entity_embedding_dim=100 --context_embedding_dim=100
python main.py --news_encoder=PNE          --user_encoder=ATT
python main.py --news_encoder=NAML         --user_encoder=ATT
python main.py --news_encoder=CNE          --user_encoder=ATT
python main.py --news_encoder=NAML_Title   --user_encoder=ATT
python main.py --news_encoder=NAML_Content --user_encoder=ATT
python main.py --news_encoder=CNE_Title    --user_encoder=ATT
python main.py --news_encoder=CNE_Content  --user_encoder=ATT</code></pre>


<hr>Ablation experiments for user encoding in Section 5.3
<pre><code>python main.py --news_encoder=CNN --user_encoder=LSTUR
python main.py --news_encoder=CNN --user_encoder=ATT
python main.py --news_encoder=CNN --user_encoder=PUE
python main.py --news_encoder=CNN --user_encoder=CATT
python main.py --news_encoder=CNN --user_encoder=MHSA
python main.py --news_encoder=CNN --user_encoder=SUE</code></pre>


<hr>Experiments for different number of GCN layers in Section 5.4
<pre><code>python main.py --news_encoder=CNE --user_encoder=SUE --gcn_layer_num=1
python main.py --news_encoder=CNE --user_encoder=SUE --gcn_layer_num=2
python main.py --news_encoder=CNE --user_encoder=SUE --gcn_layer_num=3
python main.py --news_encoder=CNE --user_encoder=SUE --gcn_layer_num=4
python main.py --news_encoder=CNE --user_encoder=SUE --gcn_layer_num=5
python main.py --news_encoder=CNE --user_encoder=SUE --gcn_layer_num=6
python main.py --news_encoder=CNE --user_encoder=SUE --gcn_layer_num=7</code></pre>