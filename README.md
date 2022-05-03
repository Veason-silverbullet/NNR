# Neural News Recommendation
This repository is for the paper [**Neural News Recommendation with Collaborative News Encoding and Structural User Encoding** (EMNLP-2021 Finding)](https://aclanthology.org/2021.findings-emnlp.5.pdf).
<br/><br/>


## Dataset Preparation
The experiments are conducted on the 200k-MIND dataset. Our code will try to download and sample the 200k-MIND dataset to the directory `../MIND-200k` (see Line 140 of `config.py` and `prepare_MIND_dataset.py`).

Since the MIND dataset is quite large, if our code cannot download it successfully due to unstable network connection, please execute the shell file `download_extract_MIND.sh` instead. If the automatic download still fails, we recommend to download the MIND dataset and knowledge graph manually according to the links in `download_extract_MIND.sh`.

Assume that now the pwd is `./NNR`, the downloaded and extracted MIND dataset should be organized as

    (terminal) $ bash download_extract_MIND.sh # Assume this command is executed successfully
    (terminal) $ cd ../MIND-200k
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
<br/>


## Environment Requirements
    (terminal) $ pip install -r requirements.txt

Our experiments require python3, torch>=1.9.0 & <=1.11.0, and torch_scatter>=2.0.9. The [torch_scatter](https://github.com/rusty1s/pytorch_scatter) package is neccessary. Our code will try to install it automatically (see Line 10 of `userEncoders.py`).
If the automatic installation fails, please follow [https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) to install the package manually.
<br/><br/>


## Experiment Running
<hr>Our Model
<pre><code>python main.py --news_encoder=CNE --user_encoder=SUE</code></pre>

<hr>Neural news recommendation baselines in Section 4.2
<pre><code>python main.py --news_encoder=DAE       --user_encoder=GRU
python main.py --news_encoder=Inception --user_encoder=CATT  --word_embedding_dim=100 --category_embedding_dim=100 --subCategory_embedding_dim=100
python main.py --news_encoder=KCNN      --user_encoder=CATT  --word_embedding_dim=100 --entity_embedding_dim=100   --context_embedding_dim=100
python main.py --news_encoder=CNN       --user_encoder=LSTUR
python main.py --news_encoder=NAML      --user_encoder=ATT
python main.py --news_encoder=PNE       --user_encoder=PUE
python main.py --news_encoder=MHSA      --user_encoder=MHSA
python main.py --news_encoder=HDC       --user_encoder=FIM   --click_predictor=FIM</code></pre>

<hr>General news recommendation baselines in Section 4.2
<pre><code>cd general_recommendation_methods
python generate_tf_idf_feature_file.py
python generate_libfm_data.py
chmod -R 777 libfm
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
<br/>


## Experiments on MIND-small and MIND-large
Experiments on MIND-small and MIND-large are available. You can specify the experiment dataset by the config parameter `--dataset=[200k,small,large] (default 200k)`.

If you would like to conduct experiments on MIND-small, please set the config parameter `--dataset=small`.

For MIND-small, we suggest the number of GCN layers of 3 and dropout rate of 0.25 (see Line 84 of `config.py`). Example command is as below:
<pre><code>python main.py --news_encoder=CNE --user_encoder=SUE --dataset=small --gcn_layer_num=3 --dropout_rate=0.25</code></pre>

If you would like to conduct experiments on MIND-large, please set the config parameter `--dataset=large`.

For MIND-large, we suggest the number of GCN layers of 4 and dropout rate of 0.1 (see Line 91 of `config.py`). Example command is as below:
<pre><code>python main.py --news_encoder=CNE --user_encoder=SUE --dataset=large --gcn_layer_num=4 --dropout_rate=0.1</code></pre>
For MIND-large, please submit the model prediction file to [*MIND leaderboard*](https://msnews.github.io/index.html#leaderboard) for performance evaluation. For example, having finished training model #1, the model prediction file is at `prediction/large/CNE-SUE/#1/prediction.zip`. If the prediction zip file is not found, please find the raw prediction file at `test/res/large/CNE-SUE/best_model_large_CNE-SUE_#1_CNE-SUE/CNE-SUE.txt`.
<br/><br/>


## Distributed Training
Distributed training is supported. If you would like to train NNR models on N GPUs, please set the config parameter `--world_size=N`. The batch size config parameter `batch_size` should be divisible by `world_size`, as our code equally divides the training batch size into N GPUs. For example,
<pre><code>python main.py --news_encoder=CNE --user_encoder=SUE --batch_size=128 --world_size=4</code></pre>
The command above trains our model on 4 GPUs, each GPU contains the mini-batch data of 32.


## Citation
```
@inproceedings{mao-etal-2021-CNE_SUE,
    title = "Neural News Recommendation with Collaborative News Encoding and Structural User Encoding",
    author = "Mao, Zhiming  and Zeng, Xingshan  and Wong, Kam-Fai",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.5",
    doi = "10.18653/v1/2021.findings-emnlp.5",
    pages = "46--55"
}
```
