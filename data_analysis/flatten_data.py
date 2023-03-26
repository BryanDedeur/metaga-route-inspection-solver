import pandas as pd 

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

df = pd.read_csv('data_export.csv', header=0, index_col=0)

flat_config = flatten_dict(config_dict, sep='.')
flat_summary = flatten_dict(summary_dict, sep='.')