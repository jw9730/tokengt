from generate_process_data.collator import collator
from generate_process_data.wrapper import SyntheticDataset
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ogb
import ogb.lsc
import ogb.graphproppred
from functools import partial

dataset = None


def get_dataset(dataset_name='', num_graphs=200, min_num_nodes=5, max_num_nodes=20, min_num_edges_attached=3,
                max_num_edges_attached=4, seed=0):
    global dataset
    if dataset is not None:
        return dataset
    dataset = {
        'num_class': 1,
        'loss_fn': F.l1_loss,
        'metric': 'mae',
        'metric_mode': 'min',
        'evaluator': ogb.lsc.PCQM4MEvaluator(),
        'dataset': SyntheticDataset(num_graphs, min_num_nodes, max_num_nodes, min_num_edges_attached,
                                    max_num_edges_attached, seed),
        'max_node': 20,
    }
    return dataset


class GraphDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str = 'Synthetic-Barabasi-Albert',
            num_workers: int = 0,
            batch_size: int = 256,
            num_graphs: int = 200,
            min_num_nodes: int = 5,
            max_num_nodes: int = 20,
            min_num_edges_attached: int = 3,
            max_num_edges_attached: int = 4,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name, num_graphs, min_num_nodes, max_num_nodes, min_num_edges_attached,
                                   max_num_edges_attached, seed)
        self.maximum_node = self.dataset['max_node']
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...

    def setup(self, stage: str = None):
        dataset = self.dataset['dataset']
        len_dataset = len(dataset)
        num_train_graph = int(0.8 * len_dataset)
        num_val_graph = int(0.1 * len_dataset)
        self.dataset_train = dataset[:num_train_graph]
        self.dataset_val = dataset[num_train_graph: num_train_graph + num_val_graph]
        self.dataset_test = dataset[num_train_graph + num_val_graph:]

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator, max_node=self.dataset['max_node']),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=self.dataset['max_node']),
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=self.dataset['max_node']),
        )
        return loader
