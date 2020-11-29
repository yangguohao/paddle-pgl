import pgl
import paddle.fluid as fluid
import numpy as np
import time
import pandas as pd

from easydict import EasyDict as edict

config = {
    "model_name": "GCNII",
    "num_layers": 5,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "edge_dropout": 0.00,
}

config = edict(config)

from collections import namedtuple

Dataset = namedtuple("Dataset", 
               ["graph", "num_classes", "train_index",
                "train_label", "valid_index", "valid_label", "test_index"])

def load_edges(num_nodes, self_loop=True, add_inverse_edge=True):
    # 从数据中读取边
    edges = pd.read_csv("work/edges.csv", header=None, names=["src", "dst"]).values

    if add_inverse_edge:
        edges = np.vstack([edges, edges[:, ::-1]])

    if self_loop:
        src = np.arange(0, num_nodes)
        dst = np.arange(0, num_nodes)
        self_loop = np.vstack([src, dst]).T
        edges = np.vstack([edges, self_loop])
    
    return edges

def load():
    # 从数据中读取点特征和边，以及数据划分
    node_feat = np.load("work/feat.npy")
    num_nodes = node_feat.shape[0]
    edges = load_edges(num_nodes=num_nodes, self_loop=True, add_inverse_edge=True)
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feat": node_feat})
    
    indegree = graph.indegree()
    norm = np.maximum(indegree.astype("float32"), 1)
    norm = np.power(norm, -0.5)
    graph.node_feat["norm"] = np.expand_dims(norm, -1)
    
    df = pd.read_csv("work/train.csv")
    node_index = df["nid"].values
    node_label = df["label"].values
    train_part = int(len(node_index) * 0.8)
    train_index = node_index[:train_part]
    train_label = node_label[:train_part]
    valid_index = node_index[train_part:]
    valid_label = node_label[train_part:]
    test_index = pd.read_csv("work/test.csv")["nid"].values
    dataset = Dataset(graph=graph, 
                    train_label=train_label,
                    train_index=train_index,
                    valid_index=valid_index,
                    valid_label=valid_label,
                    test_index=test_index, num_classes=35)
    return dataset

dataset = load()

train_index = dataset.train_index
train_label = np.reshape(dataset.train_label, [-1 , 1])
train_index = np.expand_dims(train_index, -1)

val_index = dataset.valid_index
val_label = np.reshape(dataset.valid_label, [-1, 1])
val_index = np.expand_dims(val_index, -1)

test_index = dataset.test_index
test_index = np.expand_dims(test_index, -1)
test_label = np.zeros((len(test_index), 1), dtype="int64")


import pgl
import model
import paddle.fluid as fluid
import numpy as np
import time
from build_model import build_model
#需要在paddle1.8.4版本下才能运行
# 使用CPU
#place = fluid.CPUPlace()

# 使用GPU
place = fluid.CUDAPlace(0)

train_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()
with fluid.program_guard(train_program, startup_program):
    with fluid.unique_name.guard():
        gw, loss, acc, pred = build_model(dataset,
                            config=config,
                            phase="train",
                            main_prog=train_program)

test_program = fluid.Program()
with fluid.program_guard(test_program, startup_program):
    with fluid.unique_name.guard():
        _gw, v_loss, v_acc, v_pred = build_model(dataset,
            config=config,
            phase="test",
            main_prog=test_program)


test_program = test_program.clone(for_test=True)

exe = fluid.Executor(place)
In [10]
epoch = 200
exe.run(startup_program)

# 将图数据变成 feed_dict 用于传入Paddle Excecutor
feed_dict = gw.to_feed(dataset.graph)

for epoch in range(epoch):
    # Full Batch 训练
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    feed_dict["node_index"] = np.array(train_index, dtype="int64")
    feed_dict["node_label"] = np.array(train_label, dtype="int64")
    
    train_loss, train_acc = exe.run(train_program,
                                feed=feed_dict,
                                fetch_list=[loss, acc],
                                return_numpy=True)

    # Full Batch 验证
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    feed_dict["node_index"] = np.array(val_index, dtype="int64")
    feed_dict["node_label"] = np.array(val_label, dtype="int64")
    val_loss, val_acc = exe.run(test_program,
                            feed=feed_dict,
                            fetch_list=[v_loss, v_acc],
                            return_numpy=True)
    print("Epoch", epoch, "Train Acc", train_acc[0], "Valid Acc", val_acc[0])
feed_dict["node_index"] = np.array(test_index, dtype="int64")
feed_dict["node_label"] = np.array(test_label, dtype="int64") #假标签
test_prediction = exe.run(test_program,
                            feed=feed_dict,
                            fetch_list=[v_pred],
                            return_numpy=True)[0]
print(test_prediction[0])

submission = pd.DataFrame(data={
                            "nid": test_index.reshape(-1),
                            "label": test_prediction.reshape(-1)
                        })
submission.to_csv("submission.csv", index=False)
