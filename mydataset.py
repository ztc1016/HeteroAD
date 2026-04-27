import torch
from torch import nn
import dgl
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
class MemoryEfficientDataset(Dataset):

    def __init__(self, chunks, template_csv_path, node_num, time_steps=60, metric_dim=7, topk=3):
        self.node_num = node_num
        self.time_steps = time_steps
        self.metric_dim = metric_dim
        self.topk = topk
        self.template_csv_path = template_csv_path

        self.graph_structures = []
        self.idx2id = {}

        for idx, chunk_id in enumerate(chunks.keys()):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]

            hetero_graph_structure, eventid_mapping = self._build_hetero_log_graph_structure(chunk["logs"])

            metrics = chunk.get("metrics", None)
            if metrics is not None:
                metrics = np.array(metrics, dtype=np.float32)
                if metrics.shape != (node_num, time_steps, metric_dim):
                    metrics = np.zeros((node_num, time_steps, metric_dim), dtype=np.float32)
            else:
                metrics = np.zeros((node_num, time_steps, metric_dim), dtype=np.float32)

            metrics_graphs = self._build_metrics_knn_graphs(metrics)

            self.graph_structures.append((
                hetero_graph_structure,
                metrics_graphs,
                eventid_mapping,
                chunk["culprit"]
            ))

        print(f"数据集初始化完成，共 {len(self.graph_structures)} 个样本（内存优化模式）")

    def _build_hetero_log_graph_structure(self, logs_dict):
        eventids = logs_dict.get('eventids', [])
        services = logs_dict.get('services', [])

        if len(eventids) != len(services):
            raise ValueError(f"eventids 和 services 长度不匹配")

        if len(eventids) == 0:
            return self._build_empty_hetero_graph_structure()

        nodes_by_type = {f'service_{nid}': [] for nid in range(self.node_num)}
        event_to_local_idx = {nid: {} for nid in range(self.node_num)}
        event_positions = []
        eventid_mapping = {}

        for pos, (eid, sid) in enumerate(zip(eventids, services)):
            sid = int(sid)
            node_type = f'service_{sid}'

            if eid not in event_to_local_idx[sid]:
                local_idx = len(nodes_by_type[node_type])
                event_to_local_idx[sid][eid] = local_idx

                nodes_by_type[node_type].append({
                    'event_id': eid,
                    'local_idx': local_idx
                })
                eventid_mapping[(node_type, local_idx)] = eid

            local_idx = event_to_local_idx[sid][eid]
            event_positions.append((sid, local_idx))

        edges = {}
        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            edges[(node_type, 'temporal', node_type)] = ([], [])

        for i in range(self.node_num):
            for j in range(self.node_num):
                if i != j:
                    src_type = f'service_{i}'
                    dst_type = f'service_{j}'
                    edges[(src_type, 'cross', dst_type)] = ([], [])

        for i in range(len(event_positions) - 1):
            src_sid, src_local = event_positions[i]
            dst_sid, dst_local = event_positions[i + 1]

            src_type = f'service_{src_sid}'
            dst_type = f'service_{dst_sid}'

            if src_sid == dst_sid:
                edge_key = (src_type, 'temporal', dst_type)
                if src_local != dst_local:
                    edges[edge_key][0].append(src_local)
                    edges[edge_key][1].append(dst_local)
            else:
                edge_key = (src_type, 'cross', dst_type)
                edges[edge_key][0].append(src_local)
                edges[edge_key][1].append(dst_local)

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            temporal_key = (node_type, 'temporal', node_type)
            if len(edges[temporal_key][0]) == 0 and len(nodes_by_type[node_type]) > 0:
                edges[temporal_key][0].append(0)
                edges[temporal_key][1].append(0)

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            if len(nodes_by_type[node_type]) == 0:
                nodes_by_type[node_type].append({
                    'event_id': f'EMPTY_{nid}',
                    'local_idx': 0
                })
                eventid_mapping[(node_type, 0)] = f'EMPTY_{nid}'
                edges[(node_type, 'temporal', node_type)] = ([0], [0])

        num_nodes_dict = {node_type: len(nodes) for node_type, nodes in nodes_by_type.items()}
        hetero_graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

        return hetero_graph, eventid_mapping

    def _build_empty_hetero_graph_structure(self):
        edges = {}
        nodes_by_type = {}
        eventid_mapping = {}

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            nodes_by_type[node_type] = [{
                'event_id': f'EMPTY_{nid}',
                'local_idx': 0
            }]
            eventid_mapping[(node_type, 0)] = f'EMPTY_{nid}'
            edges[(node_type, 'temporal', node_type)] = ([0], [0])

            for other_nid in range(self.node_num):
                if other_nid != nid:
                    other_type = f'service_{other_nid}'
                    edges[(node_type, 'cross', other_type)] = ([], [])

        num_nodes_dict = {node_type: len(nodes) for node_type, nodes in nodes_by_type.items()}
        hetero_graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

        return hetero_graph, eventid_mapping

    def _load_template_vectors_batch(self, event_ids):
        import csv
        template_vectors = {}
        event_ids_set = set(event_ids)

        with open(self.template_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 302:
                    event_id = row[0]
                    if event_id in event_ids_set:
                        try:
                            vector = [float(x) for x in row[-300:]]
                            template_vectors[event_id] = np.array(vector, dtype=np.float32)
                        except ValueError:
                            template_vectors[event_id] = np.zeros(300, dtype=np.float32)

        return template_vectors

    def _fill_graph_features(self, hetero_graph, eventid_mapping):
        event_ids = set(eventid_mapping.values())

        template_vectors = self._load_template_vectors_batch(event_ids)

        features_by_type = {}
        for node_type in hetero_graph.ntypes:
            num_nodes = hetero_graph.num_nodes(node_type)
            features = []
            for local_idx in range(num_nodes):
                eid = eventid_mapping.get((node_type, local_idx), f'EMPTY_{node_type.split("_")[1]}')
                feature = template_vectors.get(eid, np.zeros(300, dtype=np.float32))
                features.append(feature)
            features_by_type[node_type] = torch.FloatTensor(np.array(features))

        for node_type, features in features_by_type.items():
            hetero_graph.nodes[node_type].data['feat'] = features

        return hetero_graph

    def _build_metrics_knn_graphs(self, metrics):
        metrics_graphs = []

        for nid in range(self.node_num):
            node_metrics = metrics[nid]
            features = node_metrics.T
            g = self._construct_knn_graph(features)
            metrics_graphs.append(g)

        return metrics_graphs

    def _construct_knn_graph(self, features):
        num_vars = features.shape[0]

        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.corrcoef(features)

        dist = np.nan_to_num(dist, nan=0.0)

        src_list = []
        dst_list = []

        for i in range(num_vars):
            ind = np.argpartition(dist[i, :], -(self.topk + 1))[-(self.topk + 1):]
            for j in ind:
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)

        if len(src_list) == 0:
            for i in range(num_vars):
                src_list.append(i)
                dst_list.append(i)

        g = dgl.graph((src_list, dst_list), num_nodes=num_vars)
        g.ndata["feat"] = torch.FloatTensor(features)

        return g

    def __len__(self):
        return len(self.graph_structures)

    def __getitem__(self, idx):
        hetero_graph_structure, metrics_graphs, eventid_mapping, label = self.graph_structures[idx]

        hetero_graph = hetero_graph_structure.clone()

        hetero_graph = self._fill_graph_features(hetero_graph, eventid_mapping)

        return hetero_graph, metrics_graphs, label

    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]
class LazyGraphDataset(Dataset):

    def __init__(self, chunks, template_csv_path, node_num, time_steps=60, metric_dim=7, topk=3):
        self.node_num = node_num
        self.time_steps = time_steps
        self.metric_dim = metric_dim
        self.topk = topk

        self.template_vectors = self._load_template_vectors(template_csv_path)
        print(f"加载了 {len(self.template_vectors)} 个模板向量")

        self.raw_data = []
        self.idx2id = {}

        for idx, chunk_id in enumerate(chunks.keys()):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]

            metrics = chunk.get("metrics", None)
            if metrics is not None:
                metrics = np.array(metrics, dtype=np.float32)
                if metrics.shape != (node_num, time_steps, metric_dim):
                    print(f"警告: chunk {chunk_id} 的 metrics 形状 {metrics.shape} 不符合预期 {(node_num, time_steps, metric_dim)}")
                    metrics = np.zeros((node_num, time_steps, metric_dim), dtype=np.float32)
            else:
                metrics = np.zeros((node_num, time_steps, metric_dim), dtype=np.float32)

            self.raw_data.append((chunk["logs"], metrics, chunk["culprit"]))

        print(f"数据集初始化完成，共 {len(self.raw_data)} 个样本（懒加载模式）")

    def _load_template_vectors(self, csv_path):
        import csv
        template_vectors = {}

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 302:
                    event_id = row[0]
                    try:
                        vector = [float(x) for x in row[-300:]]
                        template_vectors[event_id] = np.array(vector, dtype=np.float32)
                    except ValueError as e:
                        print(f"警告: 解析事件 {event_id} 时出错: {e}")
                        continue

        return template_vectors

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        logs, metrics, label = self.raw_data[idx]

        hetero_log_graph = self._build_hetero_log_graph(logs)

        metrics_graphs = self._build_metrics_knn_graphs(metrics)

        return hetero_log_graph, metrics_graphs, label

    def _build_hetero_log_graph(self, logs_dict):
        eventids = logs_dict.get('eventids', [])
        services = logs_dict.get('services', [])

        if len(eventids) != len(services):
            raise ValueError(f"eventids 和 services 长度不匹配")

        if len(eventids) == 0:
            return self._build_empty_hetero_graph()

        nodes_by_type = {f'service_{nid}': [] for nid in range(self.node_num)}
        event_to_local_idx = {nid: {} for nid in range(self.node_num)}
        event_positions = []

        for pos, (eid, sid) in enumerate(zip(eventids, services)):
            sid = int(sid)
            node_type = f'service_{sid}'

            if eid not in event_to_local_idx[sid]:
                local_idx = len(nodes_by_type[node_type])
                event_to_local_idx[sid][eid] = local_idx

                feature = self.template_vectors.get(eid, np.zeros(300, dtype=np.float32))

                nodes_by_type[node_type].append({
                    'event_id': eid,
                    'feature': feature,
                    'local_idx': local_idx
                })

            local_idx = event_to_local_idx[sid][eid]
            event_positions.append((sid, local_idx))

        edges = {}
        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            edges[(node_type, 'temporal', node_type)] = ([], [])

        for i in range(self.node_num):
            for j in range(self.node_num):
                if i != j:
                    src_type = f'service_{i}'
                    dst_type = f'service_{j}'
                    edges[(src_type, 'cross', dst_type)] = ([], [])

        for i in range(len(event_positions) - 1):
            src_sid, src_local = event_positions[i]
            dst_sid, dst_local = event_positions[i + 1]

            src_type = f'service_{src_sid}'
            dst_type = f'service_{dst_sid}'

            if src_sid == dst_sid:
                edge_key = (src_type, 'temporal', dst_type)
                if src_local != dst_local:
                    edges[edge_key][0].append(src_local)
                    edges[edge_key][1].append(dst_local)
            else:
                edge_key = (src_type, 'cross', dst_type)
                edges[edge_key][0].append(src_local)
                edges[edge_key][1].append(dst_local)

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            temporal_key = (node_type, 'temporal', node_type)
            if len(edges[temporal_key][0]) == 0 and len(nodes_by_type[node_type]) > 0:
                edges[temporal_key][0].append(0)
                edges[temporal_key][1].append(0)

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            if len(nodes_by_type[node_type]) == 0:
                nodes_by_type[node_type].append({
                    'event_id': f'EMPTY_{nid}',
                    'feature': np.zeros(300, dtype=np.float32),
                    'local_idx': 0
                })
                edges[(node_type, 'temporal', node_type)] = ([0], [0])

        num_nodes_dict = {node_type: len(nodes) for node_type, nodes in nodes_by_type.items()}
        hetero_graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

        for node_type, nodes in nodes_by_type.items():
            features = np.array([node['feature'] for node in nodes])
            hetero_graph.nodes[node_type].data['feat'] = torch.FloatTensor(features)

        return hetero_graph

    def _build_empty_hetero_graph(self):
        edges = {}
        nodes_by_type = {}

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            nodes_by_type[node_type] = [{
                'event_id': f'EMPTY_{nid}',
                'feature': np.zeros(300, dtype=np.float32),
                'local_idx': 0
            }]
            edges[(node_type, 'temporal', node_type)] = ([0], [0])

            for other_nid in range(self.node_num):
                if other_nid != nid:
                    other_type = f'service_{other_nid}'
                    edges[(node_type, 'cross', other_type)] = ([], [])

        num_nodes_dict = {node_type: len(nodes) for node_type, nodes in nodes_by_type.items()}
        hetero_graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

        for node_type, nodes in nodes_by_type.items():
            features = np.array([node['feature'] for node in nodes])
            hetero_graph.nodes[node_type].data['feat'] = torch.FloatTensor(features)

        return hetero_graph

    def _build_metrics_knn_graphs(self, metrics):
        metrics_graphs = []

        for nid in range(self.node_num):
            node_metrics = metrics[nid]
            features = node_metrics.T
            g = self._construct_knn_graph(features)
            metrics_graphs.append(g)

        return metrics_graphs

    def _construct_knn_graph(self, features):
        num_vars = features.shape[0]

        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.corrcoef(features)

        dist = np.nan_to_num(dist, nan=0.0)

        src_list = []
        dst_list = []

        for i in range(num_vars):
            ind = np.argpartition(dist[i, :], -(self.topk + 1))[-(self.topk + 1):]
            for j in ind:
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)

        if len(src_list) == 0:
            for i in range(num_vars):
                src_list.append(i)
                dst_list.append(i)

        g = dgl.graph((src_list, dst_list), num_nodes=num_vars)
        g.ndata["feat"] = torch.FloatTensor(features)

        return g

    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]
class GraphDataset(Dataset):

    def __init__(self, chunks, template_csv_path, node_num, time_steps=60, metric_dim=7, topk=3):
        self.data = []
        self.idx2id = {}
        self.node_num = node_num
        self.time_steps = time_steps
        self.metric_dim = metric_dim
        self.topk = topk

        self.template_vectors = self._load_template_vectors(template_csv_path)
        print(f"加载了 {len(self.template_vectors)} 个模板向量")

        for idx, chunk_id in enumerate(chunks.keys()):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]

            hetero_log_graph = self._build_hetero_log_graph(chunk["logs"])

            metrics = chunk.get("metrics", None)
            if metrics is not None:
                metrics = np.array(metrics)
                if metrics.shape != (node_num, time_steps, metric_dim):
                    print(f"警告: chunk {chunk_id} 的 metrics 形状 {metrics.shape} 不符合预期 {(node_num, time_steps, metric_dim)}")
                    metrics = np.zeros((node_num, time_steps, metric_dim), dtype=np.float32)
            else:
                metrics = np.zeros((node_num, time_steps, metric_dim), dtype=np.float32)

            metrics_graphs = self._build_metrics_knn_graphs(metrics, use_multithreading=True, num_threads=8)

            self.data.append((hetero_log_graph, metrics_graphs, chunk["culprit"]))

    def _load_template_vectors(self, csv_path):
        import csv
        template_vectors = {}

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 302:
                    event_id = row[0]
                    try:
                        vector = [float(x) for x in row[-300:]]
                        template_vectors[event_id] = np.array(vector, dtype=np.float32)
                    except ValueError as e:
                        print(f"警告: 解析事件 {event_id} 时出错: {e}")
                        continue

        print(f"成功加载 {len(template_vectors)} 个模板向量")
        return template_vectors

    def _build_metrics_knn_graphs(self, metrics, use_multithreading=True, num_threads=8):
        from concurrent.futures import ThreadPoolExecutor
        import os

        def _build_single_graph(nid):
            node_metrics = metrics[nid]

            features = node_metrics.T

            g = self._construct_knn_graph(features, self.topk)
            return nid, g

        if use_multithreading and self.node_num > 1:
            if num_threads is None:
                num_threads = min(os.cpu_count(), self.node_num)

            metrics_graphs = [None] * self.node_num

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_build_single_graph, nid) for nid in range(self.node_num)]

                for future in futures:
                    nid, g = future.result()
                    metrics_graphs[nid] = g
        else:
            metrics_graphs = []
            for nid in range(self.node_num):
                _, g = _build_single_graph(nid)
                metrics_graphs.append(g)

        return metrics_graphs

    def _build_hetero_log_graph(self, logs_dict):
        eventids = logs_dict.get('eventids', [])
        services = logs_dict.get('services', [])

        if len(eventids) != len(services):
            raise ValueError(f"eventids 和 services 长度不匹配: {len(eventids)} vs {len(services)}")

        if len(eventids) == 0:
            return self._build_empty_hetero_graph()

        nodes_by_type = {f'service_{nid}': [] for nid in range(self.node_num)}
        event_to_local_idx = {nid: {} for nid in range(self.node_num)}
        event_positions = []

        for pos, (eid, sid) in enumerate(zip(eventids, services)):
            sid = int(sid)
            node_type = f'service_{sid}'

            if eid not in event_to_local_idx[sid]:
                local_idx = len(nodes_by_type[node_type])
                event_to_local_idx[sid][eid] = local_idx

                if eid in self.template_vectors:
                    feature = self.template_vectors[eid]
                else:
                    feature = np.zeros(300, dtype=np.float32)

                nodes_by_type[node_type].append({
                    'event_id': eid,
                    'feature': feature,
                    'local_idx': local_idx
                })

            local_idx = event_to_local_idx[sid][eid]
            event_positions.append((sid, local_idx))

        edges = {}
        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            edges[(node_type, 'temporal', node_type)] = ([], [])

        for i in range(self.node_num):
            for j in range(self.node_num):
                if i != j:
                    src_type = f'service_{i}'
                    dst_type = f'service_{j}'
                    edges[(src_type, 'cross', dst_type)] = ([], [])

        for i in range(len(event_positions) - 1):
            src_sid, src_local = event_positions[i]
            dst_sid, dst_local = event_positions[i + 1]

            src_type = f'service_{src_sid}'
            dst_type = f'service_{dst_sid}'

            if src_sid == dst_sid:
                edge_key = (src_type, 'temporal', dst_type)
                if src_local != dst_local:
                    edges[edge_key][0].append(src_local)
                    edges[edge_key][1].append(dst_local)
            else:
                edge_key = (src_type, 'cross', dst_type)
                edges[edge_key][0].append(src_local)
                edges[edge_key][1].append(dst_local)

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            temporal_key = (node_type, 'temporal', node_type)
            if len(edges[temporal_key][0]) == 0 and len(nodes_by_type[node_type]) > 0:
                edges[temporal_key][0].append(0)
                edges[temporal_key][1].append(0)

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            if len(nodes_by_type[node_type]) == 0:
                nodes_by_type[node_type].append({
                    'event_id': f'EMPTY_{nid}',
                    'feature': np.zeros(300, dtype=np.float32),
                    'local_idx': 0
                })
                edges[(node_type, 'temporal', node_type)] = ([0], [0])

        num_nodes_dict = {node_type: len(nodes) for node_type, nodes in nodes_by_type.items()}

        hetero_graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

        for node_type, nodes in nodes_by_type.items():
            features = np.array([node['feature'] for node in nodes])
            hetero_graph.nodes[node_type].data['feat'] = torch.FloatTensor(features)

        return hetero_graph

    def _build_empty_hetero_graph(self):
        edges = {}
        nodes_by_type = {}

        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            nodes_by_type[node_type] = [{
                'event_id': f'EMPTY_{nid}',
                'feature': np.zeros(300, dtype=np.float32),
                'local_idx': 0
            }]
            edges[(node_type, 'temporal', node_type)] = ([0], [0])

            for other_nid in range(self.node_num):
                if other_nid != nid:
                    other_type = f'service_{other_nid}'
                    edges[(node_type, 'cross', other_type)] = ([], [])

        num_nodes_dict = {node_type: len(nodes) for node_type, nodes in nodes_by_type.items()}
        hetero_graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

        for node_type, nodes in nodes_by_type.items():
            features = np.array([node['feature'] for node in nodes])
            hetero_graph.nodes[node_type].data['feat'] = torch.FloatTensor(features)

        return hetero_graph

    def _construct_knn_graph(self, features, topk):
        num_vars = features.shape[0]

        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.corrcoef(features)

        dist = np.nan_to_num(dist, nan=0.0)

        src_list = []
        dst_list = []

        for i in range(num_vars):
            ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]

            for j in ind:
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)

        if len(src_list) == 0:
            for i in range(num_vars):
                src_list.append(i)
                dst_list.append(i)

        g = dgl.graph((src_list, dst_list), num_nodes=num_vars)

        g.ndata["feat"] = torch.FloatTensor(features)

        return g

    def _build_log_event_graph(self, event_ids):
        if len(event_ids) == 0:
            g = dgl.graph(([], []), num_nodes=1)
            g.ndata["feat"] = torch.zeros(1, 300, dtype=torch.float32)
            return g

        unique_events = []
        event_to_idx = {}
        for eid in event_ids:
            if eid not in event_to_idx:
                event_to_idx[eid] = len(unique_events)
                unique_events.append(eid)

        num_nodes = len(unique_events)

        src_list = []
        dst_list = []

        for i in range(len(event_ids) - 1):
            src = event_to_idx[event_ids[i]]
            dst = event_to_idx[event_ids[i + 1]]
            if src != dst:
                src_list.append(src)
                dst_list.append(dst)

        if len(src_list) == 0 and num_nodes > 0:
            src_list = [0]
            dst_list = [0]

        g = dgl.graph((src_list, dst_list), num_nodes=num_nodes)

        features = []
        for eid in unique_events:
            if eid in self.template_vectors:
                features.append(self.template_vectors[eid])
            else:
                print(f"警告: 找不到事件 {eid} 的模板向量，使用零向量")
                features.append(np.zeros(300, dtype=np.float32))

        g.ndata["feat"] = torch.FloatTensor(np.array(features))

        return g

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]
class HANLogModel(nn.Module):

    def __init__(self, node_num=27, in_dim=300, hidden_dim=128, out_dim=64, num_heads=8, dropout=0.5):
        super(HANLogModel, self).__init__()

        self.node_num = node_num
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.meta_paths = self._build_meta_paths()

        from han.model_hetero import HAN

        self.han = HAN(
            meta_paths=self.meta_paths,
            in_size=in_dim,
            hidden_size=hidden_dim,
            out_size=hidden_dim,
            num_heads=[num_heads],
            dropout=dropout
        )

        self.node_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            ) for _ in range(node_num)
        ])

    def _build_meta_paths(self):
        meta_paths = []

        meta_paths.append(('temporal',))

        meta_paths.append(('cross',))

        return meta_paths

    def forward(self, hetero_graph):
        is_batch = hasattr(hetero_graph, 'batch_size') and hetero_graph.batch_size > 1
        batch_size = hetero_graph.batch_size if is_batch else 1

        h_dict = {}
        for nid in range(self.node_num):
            node_type = f'service_{nid}'
            if node_type in hetero_graph.ntypes:
                feat = hetero_graph.nodes[node_type].data['feat']

                if is_batch:
                    batch_num_nodes = hetero_graph.batch_num_nodes(node_type)

                    node_offsets = torch.cat([
                        torch.zeros(1, dtype=torch.long, device=feat.device),
                        batch_num_nodes.cumsum(0)
                    ])

                    sample_features = []
                    for i in range(batch_size):
                        start = node_offsets[i].item()
                        end = node_offsets[i + 1].item()
                        if end > start:
                            sample_feat = feat[start:end].mean(dim=0, keepdim=True)
                        else:
                            sample_feat = torch.zeros(1, feat.shape[1], device=feat.device)
                        sample_features.append(sample_feat)

                    h_dict[nid] = torch.cat(sample_features, dim=0)
                else:
                    h_dict[nid] = feat.mean(dim=0, keepdim=True)

        node_embeddings = []
        for nid in range(self.node_num):
            if nid in h_dict:
                h = h_dict[nid]
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                embed = self.node_projections[nid](h)
            else:
                if is_batch:
                    embed = torch.zeros(batch_size, self.out_dim, device=next(self.parameters()).device)
                else:
                    embed = torch.zeros(1, self.out_dim, device=next(self.parameters()).device)

            node_embeddings.append(embed)

        output = torch.stack(node_embeddings, dim=1)

        if not is_batch:
            output = output.squeeze(0)

        return output
class LogEventGraphModel(nn.Module):

    def __init__(self, in_dim=300, hidden_dim=128, out_dim=64, num_layers=2):
        super(LogEventGraphModel, self).__init__()

        from dgl.nn.pytorch import GATv2Conv

        self.layers = nn.ModuleList()

        self.layers.append(GATv2Conv(in_dim, hidden_dim, num_heads=4, allow_zero_in_degree=True))

        for _ in range(num_layers - 2):
            self.layers.append(GATv2Conv(hidden_dim * 4, hidden_dim, num_heads=4, allow_zero_in_degree=True))

        self.layers.append(GATv2Conv(hidden_dim * 4, out_dim, num_heads=1, allow_zero_in_degree=True))

        self.num_heads = 4

    def forward(self, batched_graphs_per_node):
        is_batch = hasattr(batched_graphs_per_node[0], 'batch_size') and batched_graphs_per_node[0].batch_size > 1

        node_embeddings_list = []

        for batched_g in batched_graphs_per_node:
            x = batched_g.ndata["feat"]

            for i, layer in enumerate(self.layers):
                x = layer(batched_g, x)

                if i < len(self.layers) - 1:
                    x = x.view(x.size(0), -1)
                else:
                    x = x.squeeze(1)

            batched_g.ndata['h'] = x

            graph_embeds = dgl.mean_nodes(batched_g, 'h')
            node_embeddings_list.append(graph_embeds)

        output = torch.stack(node_embeddings_list, dim=0)

        output = output.permute(1, 0, 2)

        if not is_batch:
            output = output.squeeze(0)

        return output
class MetricsGraphModel(nn.Module):

    def __init__(self, in_dim=60, hidden_dim=64, out_dim=64, num_layers=2):
        super(MetricsGraphModel, self).__init__()

        from dgl.nn.pytorch import GraphConv

        self.layers = nn.ModuleList()

        self.layers.append(GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True))

        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True))

        self.layers.append(GraphConv(hidden_dim, out_dim, allow_zero_in_degree=True))

        self.activation = nn.ReLU()

    def forward(self, batched_graphs_per_node):
        is_batch = hasattr(batched_graphs_per_node[0], 'batch_size') and batched_graphs_per_node[0].batch_size > 1

        node_embeddings_list = []

        for batched_g in batched_graphs_per_node:
            x = batched_g.ndata["feat"]

            for i, layer in enumerate(self.layers):
                x = layer(batched_g, x)
                if i < len(self.layers) - 1:
                    x = self.activation(x)

            batched_g.ndata['h'] = x

            graph_embeds = dgl.mean_nodes(batched_g, 'h')
            node_embeddings_list.append(graph_embeds)

        output = torch.stack(node_embeddings_list, dim=0)

        output = output.permute(1, 0, 2)

        if not is_batch:
            output = output.squeeze(0)

        return output
def collate_log_graphs(batch):
    batch_size = len(batch)
    node_num = len(batch[0][1])

    hetero_graphs, metrics_graph_lists, labels = zip(*batch)

    batched_hetero_graph = dgl.batch(hetero_graphs)

    def _batch_metrics_graphs(graph_lists):
        batched_graphs = []
        for node_idx in range(node_num):
            node_graphs = []
            for graph_list in graph_lists:
                g = graph_list[node_idx]
                if 'h' in g.ndata:
                    g = g.clone()
                    del g.ndata['h']
                node_graphs.append(g)

            batched_graph = dgl.batch(node_graphs)
            batched_graphs.append(batched_graph)
        return batched_graphs

    batched_metrics_graphs = _batch_metrics_graphs(metrics_graph_lists)

    batched_labels = torch.tensor(labels, dtype=torch.long)

    return batched_hetero_graph, batched_metrics_graphs, batched_labels
if __name__ == '__main__':
    import pickle

    pkl_path = '/home/ztc107552403866/WareHouse/MicroserviceDetection/A_LMDetection/Data/processed/chunks/tt/chunk_test.pkl'
    template_csv_path = '/home/ztc107552403866/WareHouse/MicroserviceDetection/A_LMDetection/Data/processed/tt/templates_300d.csv'
    node_num = 27

    print("=" * 60)
    print("开始加载数据...")
    print("=" * 60)

    with open(pkl_path, 'rb') as f:
        chunks = pickle.load(f)

    print(f"成功加载数据！样本数量: {len(chunks)}")

    print("\n" + "=" * 60)
    print("创建异构日志图数据集...")
    print("=" * 60)

    dataset = GraphDataset(chunks, template_csv_path, node_num)

    print(f"数据集大小: {len(dataset)}")

    print("\n" + "=" * 60)
    print("查看第一个样本的详情...")
    print("=" * 60)

    hetero_graph, metrics_graphs, label = dataset[0]
    chunk_id = dataset.__get_chunk_id__(0)

    print(f"\n样本索引: 0")
    print(f"Chunk ID: {chunk_id}")
    print(f"标签 (culprit): {label}")

    print("\n异构日志图信息:")
    print(f"  - 节点类型数量: {len(hetero_graph.ntypes)}")
    print(f"  - 边类型数量: {len(hetero_graph.etypes)}")
    print(f"  - 节点类型: {hetero_graph.ntypes[:5]}... (共{len(hetero_graph.ntypes)}种)")
    print(f"  - 边类型: {hetero_graph.etypes[:5]}... (共{len(hetero_graph.etypes)}种)")

    print("\n各服务节点类型的节点数量:")
    for nid in range(min(5, node_num)):
        node_type = f'service_{nid}'
        if node_type in hetero_graph.ntypes:
            num_nodes = hetero_graph.num_nodes(node_type)
            print(f"  - {node_type}: {num_nodes} 个节点")

    print("\n" + "=" * 60)
    print("Metrics KNN 图信息:")
    print("=" * 60)
    print(f"Metrics 图数量: {len(metrics_graphs)}")
    print(f"\n前3个节点的 Metrics 图详情:")
    for i in range(min(3, len(metrics_graphs))):
        g = metrics_graphs[i]
        print(f"\n  节点 {i}:")
        print(f"    - 节点数量: {g.num_nodes()} (7个变量)")
        print(f"    - 边数量: {g.num_edges()}")
        print(f"    - 节点特征形状: {g.ndata['feat'].shape} (7个变量 x 时间步)")

    print("\n" + "=" * 60)
    print("测试 HANLogModel（单样本模式）...")
    print("=" * 60)

    han_model = HANLogModel(node_num=27, in_dim=300, hidden_dim=128, out_dim=64, num_heads=8)

    with torch.no_grad():
        han_embeddings = han_model(hetero_graph)

    print(f"\nHAN 输出嵌入形状: {han_embeddings.shape}")
    print(f"预期形状: [{node_num}, 64]")
    print(f"\n前3个节点的嵌入示例:")
    for i in range(min(3, len(han_embeddings))):
        print(f"  节点 {i}: 前5维 = {han_embeddings[i][:5].numpy()}")

    print("\n" + "=" * 60)
    print("测试 HANLogModel（Batch模式）...")
    print("=" * 60)

    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_log_graphs)

    print(f"\nDataLoader 配置:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - 总样本数: {len(dataset)}")
    print(f"  - Batch 数量: {len(dataloader)}")

    batch_hetero_graph, batch_metrics_graphs, batch_labels = next(iter(dataloader))

    print(f"\n第一个 batch:")
    print(f"  - batch_hetero_graph 类型: {type(batch_hetero_graph)}")
    print(f"  - batch_metrics_graphs 长度: {len(batch_metrics_graphs)} (27个微服务节点)")
    print(f"  - batch_labels: {batch_labels}")
    if hasattr(batch_hetero_graph, 'batch_size'):
        print(f"  - batch_hetero_graph batch_size: {batch_hetero_graph.batch_size}")
    print(f"  - 第一个节点 batch 后的 metrics 图信息:")
    print(f"    * 总节点数: {batch_metrics_graphs[0].num_nodes()}")
    print(f"    * 总边数: {batch_metrics_graphs[0].num_edges()}")

    with torch.no_grad():
        batch_han_embeddings = han_model(batch_hetero_graph)

    print(f"\nHAN Batch 输出嵌入形状: {batch_han_embeddings.shape}")
    print(f"预期形状: [batch_size, node_num, out_dim] = [{batch_size}, {node_num}, 64]")

    print(f"\nHAN Batch 中第一个样本的前3个节点嵌入:")
    for i in range(min(3, batch_han_embeddings.shape[1])):
        print(f"  节点 {i}: 前5维 = {batch_han_embeddings[0][i][:5].numpy()}")

    print("\n" + "=" * 60)
    print("测试 MetricsGraphModel（单样本模式）...")
    print("=" * 60)

    metrics_model = MetricsGraphModel(in_dim=60, hidden_dim=64, out_dim=64)

    with torch.no_grad():
        metrics_embeddings = metrics_model(metrics_graphs)

    print(f"\nMetrics 输出嵌入形状: {metrics_embeddings.shape}")
    print(f"预期形状: [{node_num}, 64]")
    print(f"\n前3个节点的 metrics 嵌入示例:")
    for i in range(min(3, len(metrics_embeddings))):
        print(f"  节点 {i}: 前5维 = {metrics_embeddings[i][:5].numpy()}")

    print("\n" + "=" * 60)
    print("测试 MetricsGraphModel（Batch模式）...")
    print("=" * 60)

    with torch.no_grad():
        batch_metrics_embeddings = metrics_model(batch_metrics_graphs)

    print(f"\nMetrics Batch 输出嵌入形状: {batch_metrics_embeddings.shape}")
    print(f"预期形状: [batch_size, node_num, out_dim] = [{batch_size}, {node_num}, 64]")

    print(f"\nMetrics Batch 中第一个样本的前3个节点嵌入:")
    for i in range(min(3, batch_metrics_embeddings.shape[1])):
        print(f"  节点 {i}: 前5维 = {batch_metrics_embeddings[0][i][:5].numpy()}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)