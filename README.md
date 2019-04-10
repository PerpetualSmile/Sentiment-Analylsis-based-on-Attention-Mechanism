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
- Self-Attention & Inter-Attention BiLSTM
- Hierarchical Attention Network

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

