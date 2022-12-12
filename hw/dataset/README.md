---
dataset_info:
- config_name: coref
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: tokens
    sequence: string
  - name: ner_tags
    sequence: string
  - name: ner_spans
    sequence:
      sequence: int32
  splits:
  - name: train
    num_bytes: 6455552
    num_examples: 100
  download_size: 30074637
  dataset_size: 6455552
- config_name: entities
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: tokens
    sequence: string
  - name: ner_tags
    sequence: string
  - name: ner_spans
    sequence:
      sequence: int32
  splits:
  - name: train
    num_bytes: 6575841
    num_examples: 100
  download_size: 30074637
  dataset_size: 6575841
- config_name: events
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: tokens
    sequence: string
  - name: ner_tags
    sequence: string
  - name: ner_spans
    sequence:
      sequence: int32
  splits:
  - name: train
    num_bytes: 6466125
    num_examples: 100
  download_size: 30074637
  dataset_size: 6466125
---
