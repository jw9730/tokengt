from data import get_dataset
from lr import PolynomialDecayLR
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.functional import normalize

from models.transformer import Encoder
from models.transformer.batch_struct.sparse import Batch, make_batch_concatenated, get_mask2d, batch_like, add_null_token
from models.utils.set import to_masked_batch
from models.utils.laplacian import get_pe_2d
from models.utils.orthogonality import get_random_orthogonal_pe, gaussian_orthonormal_random_matrix
from models.label_structure import get_structure_label, get_multiple_struct_label, struct_string_map
import time
import os


class Model(pl.LightningModule):
    def __init__(self, baseline, dense_setting, attn_save_dir, n_layers, last_layer_n_heads, dim_hidden, dim_qk, dim_v,
                 dim_ff, n_heads,
                 input_dropout_rate, dropout_rate, weight_decay, dataset_name, warmup_updates, tot_updates, peak_lr,
                 end_lr,
                 list_struct_idx_str, lap_node_id=False, lap_node_id_dim=128,
                 rand_node_id=False, rand_node_id_dim=128, orf_node_id=False, orf_node_id_dim=128, type_id=False,
                 not_first_order=False,
                 maximum_node=50, save_display=False, save_display_interval=100):
        super().__init__()
        self.save_hyperparameters()
        self.dense_setting = dense_setting
        self.dim_hidden = dim_hidden
        self.lap_node_id = lap_node_id
        self.rand_node_id = rand_node_id
        self.orf_node_id = orf_node_id
        self.not_first_order = not_first_order
        self.type_id = type_id
        self.last_layer_n_heads = last_layer_n_heads
        self.save_display = save_display
        self.save_display_interval = save_display_interval
        self.encoder = Encoder(n_layers, dim_hidden, dim_hidden, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                               input_dropout_rate, dropout_rate, last_layer_n_heads=last_layer_n_heads)

        self.evaluator = get_dataset(dataset_name)['evaluator']
        self.metric = get_dataset(dataset_name)['metric']
        self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay

        self.dim_hidden = dim_hidden
        self.automatic_optimization = True
        struct_str_tmp = list_struct_idx_str.split(' ')
        self.list_struct_idx = [int(ele) for ele in struct_str_tmp]  # list(int)
        self.num_struct = len(self.list_struct_idx)
        self.maximum_node = maximum_node
        self.maximum_edges12 = 128

        self.null_token = nn.Embedding(1, dim_hidden)

        if self.lap_node_id == True:
            self.lap_node_id_dim = lap_node_id_dim
            self.laplacian_encoder = nn.Linear(self.lap_node_id_dim, dim_hidden)
        if self.type_id == True:
            self.type_embedding = nn.Embedding(2, dim_hidden)
        if self.rand_node_id == True:
            self.rand_node_id_dim = rand_node_id_dim
            self.random_encoder = nn.Linear(self.rand_node_id_dim, dim_hidden)

        if self.orf_node_id == True:
            self.orf_node_id_dim = orf_node_id_dim
            self.random_ortho_encoder = nn.Linear(orf_node_id_dim, dim_hidden)

        # save_dir for visualization 
        self.attn_save_dir = attn_save_dir
        self.attn_train_save_dir = self.attn_save_dir + '/Train'
        self.attn_val_save_dir = self.attn_save_dir + '/Val'
        self.attn_test_save_dir = self.attn_save_dir + '/Test'
        os.makedirs(self.attn_train_save_dir, exist_ok=True)
        os.makedirs(self.attn_val_save_dir, exist_ok=True)
        os.makedirs(self.attn_test_save_dir, exist_ok=True)

    def forward(self, batched_data, perturb=None):
        node_num = batched_data.node_num
        if self.dense_setting:  # dense
            edge_index = batched_data.dense_edge_index
            edge_num = batched_data.dense_edge_num
        else:  # sparse
            edge_index = batched_data.sparse_edge_index
            edge_num = batched_data.sparse_edge_num

        device = edge_index.device
        node_feature = torch.zeros(sum(node_num), self.dim_hidden, device=device)
        edge_feature = torch.zeros(sum(edge_num), self.dim_hidden, device=device)

        null_token_feature = self.null_token.weight  # [1, dim_hidden]
        G = make_batch_concatenated(node_feature, edge_index, edge_feature, node_num, edge_num,
                                    null_params={'use_null_node': False, 'null_feature': None})
        # G.values: [B, E, dim]; G.indices[B, E, 2];   
        bsize, E = G.mask.shape

        # add input embedding to graph
        # Laplacian embedding
        if self.lap_node_id:
            sparse_edge_index = batched_data.sparse_edge_index
            sparse_edge_num = batched_data.sparse_edge_num
            sparse_edge_index = sparse_edge_index.split(sparse_edge_num, 1)
            pe_list = []
            for i in range(bsize):
                pe_list.append(get_pe_2d(sparse_edge_index[i], G.indices[i], node_num[i], G.n_edges[i],
                                         half_pos_enc_dim=self.lap_node_id_dim // 2))
            pe = torch.cat(pe_list)  # [bsize, |E|, lap_node_id_dim]
            pe = self.laplacian_encoder(pe)  # [bsize, |E|, dim_hidden]
            G_pe_value = G.values + pe  # [bsize, |E|, dim_hidden]
            G = batch_like(G, G_pe_value, skip_masking=False)

        # random embedding
        if self.rand_node_id:
            if self.not_first_order:
                half_random_dim = self.rand_node_id_dim // 2
                lookup_table_random = torch.randn(self.maximum_node, half_random_dim, requires_grad=False,
                                                  device=G.device)  # [maximum_node, half_random_dim]
                lookup_table_random = normalize(input=lookup_table_random, p=2.0,
                                                dim=-1)  # [maximum_node, half_random_dim]
                random_embeddings = torch.zeros(bsize, E, 2 * half_random_dim,
                                                device=G.device)  # [bsize, E, 2*half_random_dim]
                for i in range(bsize):
                    perm = torch.randperm(self.maximum_node, device=G.device)
                    shuffle_lookup_table_random = torch.index_select(lookup_table_random, dim=0, index=perm)
                    edge12_indices_i = G.indices[i]  # [E, 2] 
                    num_edges12_i = G.n_edges[i]
                    random_embeddings[i, :num_edges12_i, :half_random_dim] = torch.index_select(
                        shuffle_lookup_table_random, 0, edge12_indices_i[:num_edges12_i, 0])
                    random_embeddings[i, :num_edges12_i, half_random_dim:] = torch.index_select(
                        shuffle_lookup_table_random, 0, edge12_indices_i[:num_edges12_i, 1])

                random_embeddings = self.random_encoder(random_embeddings)  # [bsize, E, dim_hidden]
                G_random_value = G.values + random_embeddings
                G = batch_like(G, G_random_value, skip_masking=False)

            else:
                lookup_table_random = torch.randn(self.maximum_edges12, self.rand_node_id_dim, requires_grad=False,
                                                  device=G.device)  # [maximum_edges12, rand_node_id_dim]
                lookup_table_random = normalize(input=lookup_table_random, p=2.0,
                                                dim=-1)  # [maximum_edges12, rand_node_id_dim]

                random_embeddings = torch.zeros(bsize, E, self.rand_node_id_dim,
                                                device=G.device)  # [bsize, E, rand_node_id_dim]
                for i in range(bsize):
                    num_edges12_i = G.n_edges[i]
                    perm = torch.randperm(self.maximum_edges12, device=G.device)[:num_edges12_i]
                    random_embeddings[i, :num_edges12_i, :] = torch.index_select(input=lookup_table_random, dim=0,
                                                                                 index=perm)
                random_embeddings = self.random_encoder(random_embeddings)  # [bsize, E, dim_hidden]

                G_random_value = G.values + random_embeddings
                G = batch_like(G, G_random_value, skip_masking=False)

        # type embedding
        if self.type_id:
            type_embedding = torch.zeros(bsize, E, self.dim_hidden, dtype=torch.float,
                                         device=G.device)  # [bsize, |E|, dim_hidden]
            for i in range(bsize):
                num_node_i = node_num[i]
                num_edge12_i = G.n_edges[i]
                edge_type_index = torch.ones(G.n_edges[i], dtype=torch.long, device=G.device)
                edge_type_index[:num_node_i] = torch.zeros(num_node_i, dtype=torch.long, device=G.device)
                type_emb_arr = self.type_embedding(edge_type_index)  # [num_edge12_i , dim_hidden]
                type_embedding[i, :num_edge12_i, :] = type_emb_arr
            G_type_value = G.values + type_embedding
            G = batch_like(G, G_type_value, skip_masking=False)

        # random orthogonal embedding
        if self.orf_node_id:
            if self.not_first_order:
                pe_list = []
                half_orf_node_id_dim = self.orf_node_id_dim // 2
                assert self.maximum_node <= half_orf_node_id_dim
                lookup_random_ortho_matrix = gaussian_orthonormal_random_matrix(nb_rows=self.maximum_node,
                                                                                nb_columns=half_orf_node_id_dim,
                                                                                device=G.device)  # [maximum_node, half_orf_node_id_dim]

                for i in range(bsize):
                    n_node_i = node_num[i]
                    n_edges12_i = G.n_edges[i]
                    edge12_indices = G.indices[i][:n_edges12_i]  # [n_edges12_i, 2]
                    perm = torch.randperm(self.maximum_node, device=G.device)[:n_node_i]
                    selected_rand_ortho = torch.index_select(lookup_random_ortho_matrix, dim=0,
                                                             index=perm)  # [n_node_i, half_orf_node_id_dim]
                    random_ortho_pe = torch.zeros(E, 2 * half_orf_node_id_dim, dtype=torch.float,
                                                  device=G.device)  # [E, 2*half_orf_node_id_dim]
                    random_ortho_pe[:n_edges12_i, :half_orf_node_id_dim] = torch.index_select(input=selected_rand_ortho,
                                                                                              dim=0,
                                                                                              index=edge12_indices[:,
                                                                                                    0])
                    random_ortho_pe[:n_edges12_i, half_orf_node_id_dim:] = torch.index_select(input=selected_rand_ortho,
                                                                                              dim=0,
                                                                                              index=edge12_indices[:,
                                                                                                    1])
                    pe_list.append(random_ortho_pe.unsqueeze(0))

                pe = torch.cat(pe_list)  # [bsize, |E|, orf_node_id_dim]
                pe = self.random_ortho_encoder(pe)  # [bsize, |E|, dim_hidden]
                G_pe_value = G.values + pe  # [bsize, |E|, dim_hidden]
                G = batch_like(G, G_pe_value, skip_masking=False)
            else:
                assert self.maximum_edges12 <= self.orf_node_id_dim
                random_ortho_lookup = gaussian_orthonormal_random_matrix(nb_rows=self.maximum_edges12,
                                                                         nb_columns=self.orf_node_id_dim).to(
                    G.device)  # [maximum_edges12, orf_node_id_dim]
                pe_list = []
                for i in range(bsize):
                    n_edge12_i = G.n_edges[i]
                    pe_i = torch.zeros(E, self.orf_node_id_dim, dtype=torch.float, device=G.device)
                    perm = torch.randperm(self.maximum_edges12, device=G.device)[:n_edge12_i]
                    pe_i[:n_edge12_i, :] = torch.index_select(input=random_ortho_lookup, dim=0, index=perm)
                    pe_list.append(pe_i.unsqueeze(0))
                pe = torch.cat(pe_list).to(G.device)
                pe = self.random_ortho_encoder(pe)  # [bsize, |E|, dim_hidden]
                G_pe_value = G.values + pe
                G = batch_like(G, G_pe_value, skip_masking=False)

        # add null token
        G_null = add_null_token(G, null_token_feature)
        # get attention map of model
        attn_score, output = self.encoder(G_null)  # attn_score : tensor(bsize, last_layer_n_heads, |E|+1, |E|+1);  output.values: tensor(bsize, |E|+1, dim_out)
        mask2d_G_null = get_mask2d(G, node_null=True)  # tensor((bsize, |E|, |E|+1, dtype=bool) = [128, 116, 117]
        mask2d_G = get_mask2d(G, node_null=False)  # tensor((bsize, |E|, |E|, dtype=bool)

        attn_null_mask = torch.zeros(size=(bsize, E + 1, E + 1), dtype=torch.bool,
                                     device=G.device)  # [bsize, |E|+1, |E|+1]
        attn_null_mask[:, :E, :] = mask2d_G_null  # [bsize, |E|+1, |E|+1]
        attn_null_mask = attn_null_mask.unsqueeze(1)  # [bsize, 1, |E|+1, |E|+1]
        attn_null_mask = attn_null_mask.repeat(1, self.last_layer_n_heads, 1,
                                               1)  # [bsize, last_layer_n_heads, |E|+1, |E|+1)

        attn_score = attn_score.masked_fill(attn_null_mask == False, 0)  # (bsize, last_layer_n_heads, |E|+1, |E|+1)
        attn_score = attn_score[:, :, :E, :]  # (bsize, last_layer_n_heads, |E|, |E|+1)
        struct_idx_tensor = torch.tensor(self.list_struct_idx, dtype=torch.long, device=G.device)
        # attn_score = attn_score[:, :self.num_struct, :, :]           #(bsize, num_struct, |E|, |E|+1)
        attn_score = torch.index_select(input=attn_score, dim=1,
                                        index=struct_idx_tensor)  # (bsize, num_struct, |E|, |E|+1)

        # label attention map
        multi_struct_label = get_multiple_struct_label(G, mask2d_G_null, mask2d_G,
                                                       self.list_struct_idx)  # tensor(bsize, num_struct, |E|, |E|+1)
        multi_mask2d_G_null = mask2d_G_null.unsqueeze(1).repeat(1, self.num_struct, 1,
                                                                1)  # tensor([bsize, num_struct, |E|, |E|+1])
        G_n_nodes = torch.tensor(G.n_nodes, device=G.device)  # torch(?, ?, ?, ...)
        G_n_edges = torch.tensor(G.n_edges, device=G.device)
        G_null_indices = torch.tensor(G_null.indices, device=G.device)

        return attn_score, multi_struct_label, multi_mask2d_G_null, G_n_nodes, G_n_edges, G_null_indices

    def training_step(self, batched_data, batch_idx):  # need updates
        attn_score, multi_struct_label, multi_mask2d_G_null, G_n_nodes, G_n_edges, G_null_indices = self.forward(
            batched_data)
        difference_tensor = attn_score - multi_struct_label  # [bsize, num_struct, |E|, |E|+1]
        difference_tensor = difference_tensor.masked_fill(multi_mask2d_G_null == False, 0)
        difference_tensor = torch.square(difference_tensor)  # [bsize, num_struct, |E|, |E|+1]
        bsize = difference_tensor.shape[0]
        # compare attention map leanred by model with the label attention map
        loss = torch.sum(difference_tensor, dim=(0, 1, 2, 3)) / (bsize * self.num_struct)
        struct_wise_loss = torch.sum(difference_tensor, dim=(2, 3)).mean(0)  # [num_struct,]
        struct_wise_loss = torch.log(struct_wise_loss)  # [num_struct,] 

        return {'loss': loss, 'struct_wise_loss': struct_wise_loss, 'attn_score': attn_score,
                'multi_struct_label': multi_struct_label, 'G_n_nodes': G_n_nodes, 'G_n_edges': G_n_edges,
                'G_null_indices': G_null_indices}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        avg_struct_wise_loss = torch.cat([x['struct_wise_loss'].unsqueeze(0) for x in outputs], dim=0).mean(
            0)  # [num_struct,]
        for i in range(len(self.list_struct_idx)):
            struct_idx = self.list_struct_idx[i]
            struct_string = struct_string_map[struct_idx]
            struct_loss = avg_struct_wise_loss[i]
            self.logger.experiment.add_scalars('Train_struct_wise_loss', {struct_string: struct_loss},
                                               self.current_epoch)
            self.logger.experiment.add_scalars("Loss_struct_" + struct_string, {"Train": struct_loss},
                                               self.current_epoch)

        if self.save_display:
            if (self.current_epoch + 1) % self.save_display_interval == 0:
                for i in range(len(outputs)):
                    dict_ = outputs[i]
                    attn_score = dict_['attn_score']  # tensor([bsize, num_struct, |E|, |E|+1])
                    multi_struct_label = dict_['multi_struct_label']  # tensor([bsize, num_struct, |E|, |E|+1])
                    dict_graph_info = {'G_n_nodes': dict_['G_n_nodes'], 'G_n_edges': dict_['G_n_edges'],
                                       'G_null_indices': dict_['G_null_indices'],
                                       'list_struct_idx': torch.tensor(self.list_struct_idx, device=attn_score.device)}

                    device_str = str(attn_score.device)[-1]
                    attn_score_filepath = self.attn_train_save_dir + '/Device_' + device_str + '_forward_' + str(
                        i) + '_attn.pt'
                    struct_label_filepath = self.attn_train_save_dir + '/Device_' + device_str + '_forward_' + str(
                        i) + '_struct.pt'
                    graph_info_filepath = self.attn_train_save_dir + '/Device_' + device_str + '_forward_' + str(
                        i) + '_info.pt'
                    torch.save(attn_score, attn_score_filepath)
                    torch.save(multi_struct_label, struct_label_filepath)
                    torch.save(dict_graph_info, graph_info_filepath)

    def validation_step(self, batched_data, batch_idx):
        attn_score, multi_struct_label, multi_mask2d_G_null, G_n_nodes, G_n_edges, G_null_indices = self.forward(
            batched_data)
        difference_tensor = attn_score - multi_struct_label  # [bsize, num_struct, |E|, |E|+1]
        difference_tensor = difference_tensor.masked_fill(multi_mask2d_G_null == False, 0)
        difference_tensor = torch.square(difference_tensor)  # [bsize, num_struct, |E|, |E|+1]
        bsize = difference_tensor.shape[0]
        loss = torch.sum(difference_tensor, dim=(0, 1, 2, 3)) / (bsize * self.num_struct)
        struct_wise_loss = torch.sum(difference_tensor, dim=(2, 3)).mean(0)  # [num_struct,]
        struct_wise_loss = torch.log(struct_wise_loss)  # [num_struct,] 
        return {'loss': loss, 'struct_wise_loss': struct_wise_loss, 'attn_score': attn_score,
                'multi_struct_label': multi_struct_label, 'G_n_nodes': G_n_nodes, 'G_n_edges': G_n_edges,
                'G_null_indices': G_null_indices}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)
        avg_struct_wise_loss = torch.cat([x['struct_wise_loss'].unsqueeze(0) for x in outputs], dim=0).mean(
            0)  # [num_struct,]
        for i in range(len(self.list_struct_idx)):
            struct_idx = self.list_struct_idx[i]
            struct_string = struct_string_map[struct_idx]
            struct_loss = avg_struct_wise_loss[i]
            self.logger.experiment.add_scalars('Val_struct_wise_loss', {struct_string: struct_loss}, self.current_epoch)
            self.logger.experiment.add_scalars("Loss_struct_" + struct_string, {"Val": struct_loss}, self.current_epoch)

        if self.save_display:
            if (self.current_epoch + 1) % self.save_display_interval == 0:
                for i in range(len(outputs)):
                    dict_ = outputs[i]
                    attn_score = dict_['attn_score']  # tensor([bsize, num_struct, |E|, |E|+1])
                    multi_struct_label = dict_['multi_struct_label']  # tensor([bsize, num_struct, |E|, |E|+1])
                    dict_graph_info = {'G_n_nodes': dict_['G_n_nodes'], 'G_n_edges': dict_['G_n_edges'],
                                       'G_null_indices': dict_['G_null_indices'],
                                       'list_struct_idx': torch.tensor(self.list_struct_idx, device=attn_score.device)}

                    device_str = str(attn_score.device)[-1]
                    attn_score_filepath = self.attn_val_save_dir + '/Device_' + device_str + '_forward_' + str(
                        i) + '_attn.pt'
                    struct_label_filepath = self.attn_val_save_dir + '/Device_' + device_str + '_forward_' + str(
                        i) + '_struct.pt'
                    graph_info_filepath = self.attn_val_save_dir + '/Device_' + device_str + '_forward_' + str(
                        i) + '_info.pt'
                    torch.save(attn_score, attn_score_filepath)
                    torch.save(multi_struct_label, struct_label_filepath)
                    torch.save(dict_graph_info, graph_info_filepath)

    def test_step(self, batched_data, batch_idx):
        attn_score, multi_struct_label, multi_mask2d_G_null, G_n_nodes, G_n_edges, G_null_indices = self.forward(
            batched_data)
        difference_tensor = attn_score - multi_struct_label  # [bsize, num_struct, |E|, |E|+1]
        difference_tensor = difference_tensor.masked_fill(multi_mask2d_G_null == False, 0)
        difference_tensor = torch.square(difference_tensor)  # [bsize, num_struct, |E|, |E|+1]
        bsize = difference_tensor.shape[0]
        loss = torch.sum(difference_tensor, dim=(0, 1, 2, 3)) / (bsize * self.num_struct)
        struct_wise_loss = torch.sum(difference_tensor, dim=(2, 3)).mean(0)  # [num_struct,]
        struct_wise_loss = torch.log(struct_wise_loss)  # [num_struct,] 

        return {'loss': loss, 'struct_wise_loss': struct_wise_loss, 'attn_score': attn_score,
                'multi_struct_label': multi_struct_label, 'G_n_nodes': G_n_nodes, 'G_n_edges': G_n_edges,
                'G_null_indices': G_null_indices}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("test_loss", avg_loss, sync_dist=True)
        avg_struct_wise_loss = torch.cat([x['struct_wise_loss'].unsqueeze(0) for x in outputs], dim=0).mean(
            0)  # [num_struct,]
        for i in range(len(self.list_struct_idx)):
            struct_idx = self.list_struct_idx[i]
            struct_string = struct_string_map[struct_idx]
            struct_loss = avg_struct_wise_loss[i]
            self.logger.experiment.add_scalars('Test_struct_wise_loss', {struct_string: struct_loss},
                                               self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Test_struct_" + struct_string, struct_loss, self.current_epoch)

        if self.save_display:
            for i in range(len(outputs)):
                dict_ = outputs[i]
                attn_score = dict_['attn_score']  # tensor([bsize, num_struct, |E|, |E|+1])
                multi_struct_label = dict_['multi_struct_label']  # tensor([bsize, num_struct, |E|, |E|+1])
                dict_graph_info = {'G_n_nodes': dict_['G_n_nodes'], 'G_n_edges': dict_['G_n_edges'],
                                   'G_null_indices': dict_['G_null_indices'],
                                   'list_struct_idx': torch.tensor(self.list_struct_idx, device=attn_score.device)}

                device_str = str(attn_score.device)[-1]
                attn_score_filepath = self.attn_test_save_dir + '/Device_' + device_str + '_forward_' + str(i) + '_attn.pt'
                struct_label_filepath = self.attn_test_save_dir + '/Device_' + device_str + '_forward_' + str(i) + '_struct.pt'
                graph_info_filepath = self.attn_test_save_dir + '/Device_' + device_str + '_forward_' + str(i) + '_info.pt'
                torch.save(attn_score, attn_score_filepath)
                torch.save(multi_struct_label, struct_label_filepath)
                torch.save(dict_graph_info, graph_info_filepath)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Synthetic Tokengt Transformer")
        parser.add_argument('--baseline', default=None)
        parser.add_argument('--dense_setting', action='store_true', default=False)
        parser.add_argument('--version', type=int, default=None)
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--dim_hidden', type=int, default=256)
        parser.add_argument('--dim_qk', type=int, default=256)
        parser.add_argument('--dim_v', type=int, default=256)
        parser.add_argument('--dim_ff', type=int, default=256)
        parser.add_argument('--n_heads', type=int, default=16)
        parser.add_argument('--last_layer_n_heads', type=int, default=16)
        parser.add_argument('--input_dropout_rate', type=float, default=0.)
        parser.add_argument('--dropout_rate', type=float, default=0.)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=1e-4)
        parser.add_argument('--end_lr', type=float, default=1e-4)
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--profile', action='store_true', default=False)
        parser.add_argument('--list_struct_idx_str', type=str, default="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15")
        parser.add_argument('--lap_node_id', action='store_true', default=False)
        parser.add_argument('--lap_node_id_dim', type=int, default=128)
        parser.add_argument('--rand_node_id', action='store_true', default=False)
        parser.add_argument('--rand_node_id_dim', type=int, default=128)
        parser.add_argument('--orf_node_id', action='store_true', default=False)
        parser.add_argument('--orf_node_id_dim', type=int, default=128)
        parser.add_argument('--type_id', action='store_true', default=False)
        parser.add_argument('--not_first_order', action='store_true', default=False)
        parser.add_argument('--save_display', action='store_true', default=False)
        parser.add_argument('--save_display_interval', type=int, default=50)
        return parent_parser
