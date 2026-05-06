from __future__ import annotations
import os
import json


def get_list_from_file(filename):
    l = []
    with open(filename, 'r') as file:
        for line in file:
            l.append(line.strip().replace(',','').lower())
    return l.copy()

def construct_path(*args):
    return os.path.join(*args)

def get_all_X(columtype,df):
    return [x for x in df.columns.tolist() if columtype in x.lower()]


def subset_safe(ts, wanted):
    available = set(ts.components)
    return ts[[c for c in wanted if c in available]]

def parse_lstm_variant(variant: str) -> tuple[str, int]:
    """'lstm_w14' -> ('LSTM', 14),  'gru_w7' -> ('GRU', 7)"""
    parts = variant.split("_")
    model_type = parts[0].upper()
    icl = int(parts[-1].lstrip("w"))
    return model_type, icl