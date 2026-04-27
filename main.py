from torch.utils.data import Dataset, DataLoader, random_split
from mydataset import GraphDataset, LazyGraphDataset, MemoryEfficientDataset, collate_log_graphs
import torch
import dgl
import os
import sys
CUDA_DEVICE_ID = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE_ID
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils_function import *
from base import BaseModel
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=42, type=int, help="随机种子，保证实验可复现")
parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true", help="是否使用GPU训练")
parser.add_argument("--epoches", default=50, type=int, help="训练轮数")
parser.add_argument("--batch_size", default=512, type=int, help="每批次样本数量")
parser.add_argument("--lr", default=0.001, type=float, help="优化器学习率")
parser.add_argument("--patience", default=10, type=int, help="早停耐心值")
parser.add_argument("--self_attn", default=True, type=lambda x: x.lower() == "true", help="是否使用自注意力机制")
parser.add_argument("--fuse_dim", default=128, type=int, help="特征融合后的维度")
parser.add_argument("--alpha", default=0.5, type=float, help="特征融合权重参数")
parser.add_argument("--locate_hiddens", default=[64], type=int, nargs='+', help="故障定位隐藏层维度")
parser.add_argument("--detect_hiddens", default=[64], type=int, nargs='+', help="故障检测隐藏层维度")
parser.add_argument("--log_dim", default=16, type=int, help="日志特征维度")
parser.add_argument("--trace_kernel_sizes", default=[2], type=int, nargs='+', help="链路追踪卷积核大小")
parser.add_argument("--trace_hiddens", default=[64], type=int, nargs='+', help="链路追踪隐藏层维度")
parser.add_argument("--metric_kernel_sizes", default=[2], type=int, nargs='+', help="指标卷积核大小")
parser.add_argument("--metric_hiddens", default=[64], type=int, nargs='+', help="指标隐藏层维度")
parser.add_argument("--graph_hiddens", default=[64], type=int, nargs='+', help="图神经网络隐藏层维度")
parser.add_argument("--attn_head", default=4, type=int, help="注意力头数，用于GAT或GAT-v2")
parser.add_argument("--activation", default=0.2, type=float, help="LeakyReLU激活函数的负斜率，应在(0,1)范围内")
parser.add_argument("--data", type=str, default="sn", help="数据集名称")
parser.add_argument("--result_dir", default="../result/", help="实验结果保存目录")
parser.add_argument("--dataset_mode", default="eager", type=str,
                    choices=["memory_efficient", "lazy", "eager"],
                    help="数据集加载模式: memory_efficient=预构建图+CSV读特征(推荐), lazy=动态构建图, eager=传统预加载")
params = vars(parser.parse_args())
import logging
def get_device(gpu):
    if gpu and torch.cuda.is_available():
        logging.info(f"Using GPU (CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID})...")
        return torch.device("cuda:0")
    logging.info("Using CPU...")
    return torch.device("cpu")

def run(evaluation_epoch=10):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "..", "Data", "processed", "chunks", params["data"])
    metadata = read_json(os.path.join(data_dir, "metadata.json"))
    event_num, node_num, metric_num = metadata["event_num"], metadata["node_num"], metadata["metric_num"]
    params["chunk_lenth"] = metadata["chunk_lenth"]
    edges = (metadata["edges"][0], metadata["edges"][1])
    hash_id = dump_params(params)
    params["hash_id"] = hash_id

    seed_everything(params["random_seed"])
    device = get_device(params["gpu"])
    train_chunks, test_chunks = load_chunks(data_dir)
    template_csv_path = os.path.join(data_dir, "templates_300d.csv")
    dataset_mode = params.get("dataset_mode", "memory_efficient")
    if dataset_mode == "memory_efficient":
        logging.info("使用内存优化模式（MemoryEfficientDataset）- 预构建图结构，从CSV动态读取特征")
        DatasetClass = MemoryEfficientDataset
    elif dataset_mode == "lazy":
        logging.info("使用懒加载模式（LazyGraphDataset）- 动态构建图结构")
        DatasetClass = LazyGraphDataset
    else:
        logging.info("使用传统模式（GraphDataset）- 预加载所有数据到内存")
        DatasetClass = GraphDataset

    chunk_lenth = metadata["chunk_lenth"]
    full_train_data = DatasetClass(train_chunks, template_csv_path, node_num, time_steps=chunk_lenth)
    test_data = DatasetClass(test_chunks, template_csv_path, node_num, time_steps=chunk_lenth)
    train_size = int(0.8 * len(full_train_data))
    valid_size = len(full_train_data) - train_size
    train_data, valid_data = random_split(
        full_train_data,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(params["random_seed"])
    )
    logging.info(f"数据集分割: 训练集 {train_size}, 验证集 {valid_size}, 测试集 {len(test_data)}")
    train_dl = DataLoader(
        train_data,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_log_graphs,
        pin_memory=True
    )
    valid_dl = DataLoader(
        valid_data,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_log_graphs,
        pin_memory=True
    )
    test_dl = DataLoader(
        test_data,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_log_graphs,
        pin_memory=True
    )
    params['metrics_in_dim'] = chunk_lenth

    model = BaseModel(event_num, metric_num, node_num, device, **params)

    scores, converge = model.fit(train_dl, valid_dl, test_dl, evaluation_epoch=evaluation_epoch)
    result_dir = os.path.join(base_dir, "..", params["result_dir"].lstrip("./").lstrip("../"))
    os.makedirs(result_dir, exist_ok=True)

    dump_scores(result_dir, hash_id, scores, converge)
    logging.info("Current hash_id {}".format(hash_id))
    model_filename = f"{params['data']}_model.pt"
    model_path = os.path.join(result_dir, model_filename)
    torch.save(model.model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
if "__main__" == __name__:
    run()