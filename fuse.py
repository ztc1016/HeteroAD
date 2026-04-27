import torch
import torch.nn as nn
import dgl
from mydataset import LogEventGraphModel, MetricsGraphModel
class CrossAttention(nn.Module):

    def __init__(self, dimensions):
        super(CrossAttention, self).__init__()
        self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.tanh = nn.Tanh()
    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        query = query.reshape(batch_size * output_len, dimensions)
        query = self.linear_in(query)
        query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)
        return output, attention_weights
class GraphFusionEncoder(nn.Module):

    def __init__(self, hidden_size=64, node_num=27, fuse_type="cross_attn"):
        super(GraphFusionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.node_num = node_num
        self.fuse_type = fuse_type

        if self.fuse_type == "cross_attn" or self.fuse_type == "sep_attn":
            self.attn_alpha = CrossAttention(self.hidden_size)
            self.attn_beta = CrossAttention(self.hidden_size)

    def forward(self, log_features, metrics_features):
        fused = None

        if self.fuse_type == "cross_attn":
            fused_alpha, _ = self.attn_alpha(query=log_features, context=metrics_features)

            fused_beta, _ = self.attn_beta(query=metrics_features, context=log_features)

            fused = torch.cat((fused_alpha, fused_beta), dim=1)

        elif self.fuse_type == "sep_attn":
            fused_metrics, _ = self.attn_alpha(query=metrics_features, context=metrics_features)

            fused_log, _ = self.attn_beta(query=log_features, context=log_features)

            fused = torch.cat((fused_metrics, fused_log), dim=1)

        elif self.fuse_type == "concat":
            fused = torch.cat((log_features, metrics_features), dim=1)

        return fused, (log_features, metrics_features)
class GraphFusionModel(nn.Module):

    def __init__(self, log_model, metrics_model, hidden_size=64, num_classes=2,
                 fuse_type="cross_attn", dropout=0.1):
        super(GraphFusionModel, self).__init__()

        self.log_model = log_model
        self.metrics_model = metrics_model
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fuse_type = fuse_type
        self.node_num = None

        self.fusion_encoder = None

        self.classifier = None
        self.dropout = dropout

        self.criterion = nn.CrossEntropyLoss()

    def _build_classifier(self, node_num):
        if self.fuse_type in ["cross_attn", "sep_attn", "concat"]:
            fused_len = node_num + node_num
        else:
            fused_len = node_num

        self.classifier = nn.Sequential(
            nn.Linear(fused_len * self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, log_graphs, metrics_graphs, labels=None, flag=False):
        node_num = len(log_graphs)

        if self.node_num is None:
            self.node_num = node_num
            self.fusion_encoder = GraphFusionEncoder(
                hidden_size=self.hidden_size,
                node_num=node_num,
                fuse_type=self.fuse_type
            )
            self._build_classifier(node_num)
            if hasattr(log_graphs[0], 'device'):
                device = log_graphs[0].device
                self.fusion_encoder = self.fusion_encoder.to(device)
                self.classifier = self.classifier.to(device)

        log_features = self.log_model(log_graphs)

        metrics_features = self.metrics_model(metrics_graphs)

        if log_features.dim() == 2:
            log_features = log_features.unsqueeze(0)
        if metrics_features.dim() == 2:
            metrics_features = metrics_features.unsqueeze(0)

        fused_features, (log_re, metrics_re) = self.fusion_encoder(log_features, metrics_features)

        batch_size = fused_features.size(0)
        fused_flat = fused_features.view(batch_size, -1)

        logits = self.classifier(fused_flat)

        if flag:
            y_pred = logits.detach().cpu().numpy().argmax(axis=1)
            conf = torch.softmax(logits, dim=1).detach().cpu().numpy().max(axis=1)
            return {"y_pred": y_pred, "conf": conf}

        if labels is not None:
            loss = self.criterion(logits, labels.long())
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
if __name__ == "__main__":
    import sys
    sys.path.append('/home/ztc107552403866/WareHouse/MicroserviceDetection/A_LMDetection2/src')

    print("=" * 60)
    print("测试 GraphFusionModel")
    print("=" * 60)

    log_model = LogEventGraphModel(in_dim=300, hidden_dim=128, out_dim=64)
    metrics_model = MetricsGraphModel(in_dim=60, hidden_dim=64, out_dim=64)

    fusion_model = GraphFusionModel(
        log_model=log_model,
        metrics_model=metrics_model,
        hidden_size=64,
        num_classes=2,
        fuse_type="cross_attn"
    )

    print(f"\n模型创建成功！")
    print(f"融合类型: cross_attn")

    batch_size = 4

    log_features = torch.randn(batch_size, 25, 64)

    metrics_features = torch.randn(batch_size, 25, 64)

    labels = torch.randint(0, 2, (batch_size,))

    print(f"\n输入形状:")
    print(f"  日志特征: {log_features.shape}")
    print(f"  指标特征: {metrics_features.shape}")
    print(f"  标签: {labels.shape}")

    fusion_encoder = GraphFusionEncoder(hidden_size=64, fuse_type="cross_attn")
    fused, (log_re, metrics_re) = fusion_encoder(log_features, metrics_features)

    print(f"\n融合编码器输出:")
    print(f"  融合特征: {fused.shape}")
    print(f"  日志特征: {log_re.shape}")
    print(f"  指标特征: {metrics_re.shape}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)