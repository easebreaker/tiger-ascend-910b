Third-Party Notices
===================

This repository adapts code and ideas from:

1) Datawhale torch-rechub
   https://github.com/datawhalechina/torch-rechub
   License: Apache License 2.0
   Components adapted:
     - TIGERModel (T5ForConditionalGeneration subclass with temperature CE)
     - TigerSeqDataset / Trie constrained decoding utilities
     - Toy MovieLens-style semantic-ID workflow

2) Hugging Face Transformers (T5)
   https://github.com/huggingface/transformers
   License: Apache License 2.0

3) Amazon Beauty processed interactions + RQ-VAE semantic index
   Source packaging: https://github.com/NonameUntitled/tiger
   (see data/amazon_beauty/SOURCE_LICENCE)
   Original reviews: https://jmcauley.ucsd.edu/data/amazon/
   Paper: Rajput et al., Recommender Systems with Generative Retrieval (NeurIPS 2023)

Paper reference:
  Rajput et al., Recommender Systems with Generative Retrieval (TIGER).
