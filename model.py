import torch
from torch import nn
from dgl.nn.pytorch import GATv2Conv
from dgl.nn import GlobalAttentionPooling
from fuse import GraphFusionEncoder
from mydataset import HANLogModel, MetricsGraphModel
class FullyConnected(nn.Module):

    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []

        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]

        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor):
        return self.net(x)
import numpy as np
class MainModel(nn.Module):

    def __init__(self, event_num, metric_num, node_num, device, alpha=0.5, debug=False, **kwargs):
        super(MainModel, self).__init__()
        self.device = device
        self.node_num = node_num
        self.alpha = alpha
        hidden_size = kwargs.get('hidden_size', 64)
        fuse_type = kwargs.get('fuse_type', 'cross_attn')
        self.log_encoder = HANLogModel(
            node_num=node_num,
            in_dim=kwargs.get('log_in_dim', 300),
            hidden_dim=kwargs.get('log_hidden_dim', 128),
            out_dim=hidden_size,
            num_heads=kwargs.get('log_num_heads', 8),
            dropout=kwargs.get('log_dropout', 0.5)
        ).to(device)
        self.metrics_encoder = MetricsGraphModel(
            in_dim=kwargs.get('metrics_in_dim', 60),
            hidden_dim=kwargs.get('metrics_hidden_dim', 64),
            out_dim=hidden_size,
            num_layers=kwargs.get('metrics_num_layers', 2)
        ).to(device)
        self.fusion_encoder = GraphFusionEncoder(hidden_size=hidden_size, node_num=node_num, fuse_type=fuse_type)
        self.fusion_encoder.to(device)
        feat_out_dim = node_num * 2 * hidden_size
        self.detector = FullyConnected(feat_out_dim, 2,
                                       kwargs['detect_hiddens']).to(device)
        self.detector_criterion = nn.CrossEntropyLoss()

        self.localizer = FullyConnected(feat_out_dim, node_num,
                                        kwargs['locate_hiddens']).to(device)
        self.localizer_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.get_prob = nn.Softmax(dim=-1)
    def forward(self, log_graphs, metrics_graphs, fault_indexs):
        log_features = self.log_encoder(log_graphs)
        metrics_features = self.metrics_encoder(metrics_graphs)
        if log_features.dim() == 2:
            log_features = log_features.unsqueeze(0)
        if metrics_features.dim() == 2:
            metrics_features = metrics_features.unsqueeze(0)
        batch_size = log_features.size(0)
        fused_features, _ = self.fusion_encoder(log_features, metrics_features)
        embeddings = fused_features.view(batch_size, -1)

        y_prob = torch.zeros((batch_size, self.node_num)).to(self.device)
        for i in range(batch_size):
            if fault_indexs[i] > -1:
                y_prob[i, fault_indexs[i]] = 1

        y_anomaly = torch.zeros(batch_size).long().to(self.device)
        for i in range(batch_size):
            y_anomaly[i] = int(fault_indexs[i] > -1)
        locate_logits = self.localizer(embeddings)
        locate_loss = self.localizer_criterion(locate_logits, fault_indexs.to(self.device))

        detect_logits = self.detector(embeddings)
        detect_loss = self.detector_criterion(detect_logits, y_anomaly)

        loss = self.alpha * detect_loss + (1 - self.alpha) * locate_loss
        node_probs = self.get_prob(locate_logits.detach()).cpu().numpy()

        y_pred = self.inference(batch_size, node_probs, detect_logits)

        return {
            'loss': loss,
            'y_pred': y_pred,
            'y_prob': y_prob.detach().cpu().numpy(),
            'pred_prob': node_probs
        }

    def inference(self, batch_size, node_probs, detect_logits=None):
        node_list = np.flip(node_probs.argsort(axis=1), axis=1)

        y_pred = []
        for i in range(batch_size):
            detect_pred = detect_logits.detach().cpu().numpy().argmax(axis=1).squeeze()

            if detect_pred[i] < 1:
                y_pred.append([-1])
            else:
                y_pred.append(node_list[i])

        return y_pred