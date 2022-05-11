import torch
import numpy as np
# np.set_printoptions(suppress=True)

np.set_printoptions(threshold=np.inf)
from sklearn.metrics import roc_auc_score

from transformers import BertModel, BertConfig, BertTokenizer

import torch_geometric.transforms as T
from torch_geometric.transforms import ToUndirected, RandomLinkSplit, NormalizeFeatures,ToDevice
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv,RGCNConv,GCN2Conv,GENConv,FiLMConv,SuperGATConv,BatchNorm
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
modelConfig = BertConfig.from_pretrained('bert-base-uncased')
textExtractor1 = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainpath = "./ATGdataset/train/"

x = torch.tensor(allw2v).type(torch.float).to(device)
edge_index = torch.tensor([start_edge_id,end_edge_id]).type(torch.long).to(device)
neg_edge_index = torch.tensor([start_edge_id_0,end_edge_id_0]).type(torch.long).to(device)


label_l = [1.]*len(start_edge_id)
label = torch.tensor(label_l).type(torch.float).to(device)
# print(label)

traindata = Data(x=x,edge_index=edge_index,neg_edge_index=neg_edge_index,edge_label_index = edge_index,edge_label= label)

transform = RandomLinkSplit(
    num_val=0,
    num_test=0,
    is_undirected=True,
    add_negative_train_samples=False
)

ToDevice(device)
NormalizeFeatures(traindata)
#
train_data = traindata


x1 = torch.tensor(test_allw2v).type(torch.float).to(device)
# edge_index = torch.tensor([a_s_e_id,a_e_e_id])
edge_index1 = torch.tensor([test_start_edge_id,test_end_edge_id]).type(torch.long).to(device)
neg_edge_index1 = torch.tensor([test_start_edge_id_0,test_end_edge_id_0]).type(torch.long).to(device)

label_l_test = [1.]*len(test_start_edge_id)
label_test = torch.tensor(label_l_test).type(torch.float).to(device)
# print(label)

transform = RandomLinkSplit(
    num_val=0,
    num_test=0,
    is_undirected=True,
    add_negative_train_samples=False
)

testdata = Data(x=x1,edge_index=edge_index1,neg_edge_index=neg_edge_index1, edge_label_index = edge_index1,edge_label= label_test)
NormalizeFeatures(testdata)

test_data = testdata
val_data = test_data



class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
    # def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
     

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)

        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = Net(traindata.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train(epoch):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    a = train_data.x[1]
    b = train_data.edge_index

    edge_label_index = torch.cat(
        [train_data.edge_label_index, train_data.neg_edge_index],
        dim=-1,
    )

    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)


    # for key in train_data.keys:
    #     print(key, getattr(train_data, key).shape)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss

