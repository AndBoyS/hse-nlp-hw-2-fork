from pathlib import Path
from typing import *
from collections import defaultdict
import re

import pandas as pd


class LitbankDataset:

    label_to_columns = {
        'entities': ['token'] + [f'label{i}' for i in range(5)],
        'events': ['token', 'label'],
    }
    # {тип разметки: словарь {название файла: датафрейм с разметкой}}
    df_dicts: Dict[str, Dict[str, pd.DataFrame]] = {}
    # {тип разметки: объединенный датафрейм}
    dfs: Dict[str, pd.DataFrame] = {}

    def __init__(self, repo_dir: Path):
        self.label_to_dir = {label: repo_dir / label for label in self.label_to_columns}

        self._load_raw_texts()
        for label in self.label_to_columns:
            self._load_label(label)

    def _load_raw_texts(self):
        # {название файла: сырой текст}
        raw_texts_dir = self.label_to_dir['entities'] / 'brat'
        self.raw_texts_dict = {fp.name: fp.read_text()
                               for fp in raw_texts_dir.glob('*.txt')}

    def _load_label(self, label_type: str):
        assert label_type in self.label_to_dir.keys()
        label_dir = self.label_to_dir[label_type] / 'tsv'
        df_dict = {fp.name: self.read_litbank_tsv(fp, label_type)
                   for fp in label_dir.glob('*.tsv')}
        self.df_dicts[label_type] = df_dict
        self.dfs[label_type] = pd.concat(df_dict.values()).fillna(0)

    def read_litbank_tsv(self, fp: Path, label_type: str) -> pd.DataFrame:
        """
        Загрузить .tsv файл из датасета litbank в формате pandas.Dataframe
        """

        text = fp.read_text()
        data_list = [s.rstrip().split('\t') for s in text.split('\n')
                     if s]

        df = pd.DataFrame(data_list)
        cols = self.label_to_columns[label_type]
        cols = cols[:df.shape[1]]
        df.columns = cols

        return df

    def __len__(self):
        return len(self.raw_texts_dict)

    def __str__(self):
        return f'LitbankDataset({len(self)})'

    def __repr__(self):
        return str(self)