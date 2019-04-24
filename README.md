# Sentiment-Analylsis-based-on-Attention-Mechanism

## Introduce

- We tried various attention models on sentiment analysis task, such as InterAttention-BiLSTM, Transformer(Self-Attention), Self-Attention&Inter-Attention-BiLSTM, HAN.

- We proposed TransformerForClassification model which only needs attention mechanism and does not contain any RNN architecture.
- We trained and tested our models on both English and Chinese sentiment analysis dataset.

- We intuitively proved the reasonability and power of attention mechanism by attention visualization.

- We crawled our own Chinese movie review dataset and made it public.

## Model implemented in this repository

- Inter-Attention BiLSTM 
> Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).
- Transformer for classification
> Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.
- Self-Attention & Inter-Attention BiLSTM
- Hierarchical Attention Network
> Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016.

## Model Architecture
- Inter-Attention BiLSTM

![](model&#32;graph/Attention-BiLSTM.png)

- Transformer

![](model&#32;graph/Transformer.png)

- Self-Attention & Inter-Attention BiLSTM

![](model&#32;graph/SI-Attention-BiLSTM.png)

- Hierarchical Attention Network

![](model&#32;graph/HAN.png)


## Attention Visualization

![](attention&#32;visualization/Douban/Attention-h-query-BiLSTM/Attention-h-query-BiLSTM-on-Douban-90000.png)

![](attention&#32;visualization/Douban/Attention-h-query-BiLSTM/Attention-h-query-BiLSTM-on-Douban-67.png)

![](attention&#32;visualization/Douban/Attention-h-query-BiLSTM/Attention-h-query-BiLSTM-on-Douban-51306.png)

![](attention&#32;visualization/Douban/Attention-h-query-BiLSTM/Attention-h-query-BiLSTM-on-Douban-38.png)

![](attention&#32;visualization/Yelp&#32;Polarity/Attention-h-query-BiLSTM/Attention-h-query-BiLSTM-on-Yelp-Polarity-11.png)

![](attention&#32;visualization/Yelp&#32;Polarity/Attention-h-query-BiLSTM/Attention-h-query-BiLSTM-on-Yelp-Polarity-100.png)

## Hierarchical Attention Visualization

![](attention&#32;visualization/Douban/Hierarchical/Hierarchical-Attention-Networks-on-Douban-38.png)

![](attention&#32;visualization/Douban/Hierarchical/Hierarchical-Attention-Networks-on-Douban-46.png)


![](attention&#32;visualization/Douban/Hierarchical/Hierarchical-Attention-Networks-on-Douban-67.png)

