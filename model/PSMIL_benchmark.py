import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
import warnings

# 忽略ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)



F_DIM = 512


class Backbone(nn.Module):
    def __init__(self, dataset="MNIST"):
        super(Backbone, self).__init__()

        self.stem = None

        dim_orig = 0
        if dataset == "MUSK1" or dataset == "MUSK2":
            dim_orig = 167
        elif dataset == "TIGER" or dataset == "FOX" or dataset == "ELEPHANT":
            dim_orig = 231


        self.stem = nn.Sequential(
            nn.Linear(dim_orig, F_DIM),
            nn.ReLU(),
            nn.Linear(F_DIM, F_DIM),
            nn.ReLU(),
            # [t,32,4,4]
        )

    def forward(self, x):
        x = self.stem(x)
        return x.view(x.shape[0], -1)  # [t,F_DIM]

class GatedAttentionLayerV(nn.Module):
    '''
    $\text{tanh}\left(\boldsymbol{W}_{v}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_v \right)$ in Equation (2)
    '''

    def __init__(self, dim=512):
        super(GatedAttentionLayerV, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_V, b_V):
        out = F.linear(features, W_V, b_V)
        out_tanh = torch.tanh(out)

        return out_tanh

class GatedAttentionLayerU(nn.Module):
    '''
    $\text{sigm}\left(\boldsymbol{W}_{u}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_u \right)$ in Equation (2)
    '''

    def __init__(self, dim=512):
        super(GatedAttentionLayerU, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_U, b_U):
        out = F.linear(features, W_U, b_U)
        out_sigmoid = torch.sigmoid(out)

        return out_sigmoid

class MILModel(nn.Module):
    def __init__(self, device, dataset="MNIST", pooling="psa"):
        super(MILModel, self).__init__()

        self.classifiers = None
        self.channels = 100
        self.stem = Backbone(dataset)
        self.dataset = dataset
        self.clsnum = 2
        self.pooling = pooling
        self.attention = nn.Sequential(
            nn.Linear(F_DIM, F_DIM),
            nn.ReLU(),
            nn.Linear(F_DIM, 1),
        )
        self.ps_attention = nn.Sequential(
            nn.Linear(self.clsnum, 1),
        )

        self.att_layer_V = GatedAttentionLayerV(F_DIM)
        self.att_layer_U = GatedAttentionLayerU(F_DIM)
        self.linear_V = nn.Linear(F_DIM, self.clsnum)
        self.linear_U = nn.Linear(F_DIM, self.clsnum)

        self.tpt = TransMIL( self.clsnum).to(device)
        self.linear = nn.Parameter(data=torch.FloatTensor(F_DIM, 2), requires_grad=True).to(device)
        self.firsttime = True
        self.fbank = nn.Parameter(data=torch.FloatTensor(F_DIM, self.clsnum), requires_grad=False)

        if dataset == "messidor":
            self.linear = nn.Parameter(data=torch.FloatTensor(2048, 2)).to(device)
        nn.init.kaiming_uniform_(self.linear)
        self.softmax = nn.Softmax(dim=0)

        dim_orig = 0
        if dataset == "MUSK1" or dataset == "MUSK2":
            dim_orig = 167
        elif dataset == "TIGER" or dataset == "FOX" or dataset == "ELEPHANT":
            dim_orig = 231
       


        self.stem = nn.Sequential(
            nn.Linear(dim_orig, F_DIM),
            nn.ReLU(),
            nn.Linear(F_DIM, F_DIM),
            nn.ReLU(),
            # [t,32,4,4]
        )

    def forward(self, x,y, epoch):
        fs = self.stem(x)  # [t,o]
        if self.pooling == "psa":
            if self.firsttime:
                self.fbank.data = fs.mean(dim=0, keepdim=True).repeat(self.clsnum, 1).transpose(0, 1)
                self.firsttime = False

            with torch.no_grad():
                pred = torch.softmax(torch.mm(fs, self.fbank), dim=1)  # [t,ks]
                predicted_labels = torch.argmax(pred, dim=1)  # Find the predicted class for each sample
                pred_one_hot = nn.functional.one_hot(predicted_labels, num_classes=pred.shape[1]).float()

            alpha = self.ps_attention(pred)  # [t,1]
            alpha1 = self.ps_attention(pred_one_hot)  # [t,1]
            alpha = torch.softmax(alpha.T, dim=1)  # [1,t]
            alpha1 = torch.softmax(alpha1.T, dim=1)  # [1,t]

            alpha = (alpha + alpha) / 2

            F = torch.mm(alpha, fs)  # [1,o]
            selected_samples = (predicted_labels == y)

            if torch.any(selected_samples):
                selected_indices = torch.nonzero(selected_samples).squeeze()  # 获取为True的索引
                critical_fs = fs[selected_indices]  # 获取关键特征
                criticalF = torch.mean(critical_fs, dim=0, keepdim=True)
            else:
                criticalF = F

            if y >= 0:
                newf = nn.functional.normalize((0.999 * self.fbank[:, y]) + (0.001 * criticalF), dim=0)
                self.fbank[:, y] = newf

            if torch.isnan(self.fbank).any():
                print("Tensor Fbank contains NaN values")
        if self.pooling == "fsa":
            alpha = self.attention(fs)  # [t,1]
            alpha = torch.softmax(alpha.T, dim=1)  # [1,t]

            F = torch.mm(alpha, fs)  # [1,o]

        if self.pooling == "mha":
            A_V = self.att_layer_V(fs, self.linear_V.weight, self.linear_V.bias)
            A_U = self.att_layer_U(fs, self.linear_U.weight, self.linear_U.bias)
            A = self.ps_attention(A_V * A_U)
            # alpha = torch.softmax(A.T, dim=1)  # [1,t]
            alpha = torch.transpose(A, 1, 0)
            alpha = torch.sigmoid(alpha)
            c = torch.sum(alpha)
            alpha = alpha/c
            F = torch.mm(alpha, fs)  # Equation (3)

        if self.pooling == "tpt":
            Y_logits = self.tpt(fs.unsqueeze(0)).squeeze()
            Y_hat = torch.argmax(Y_logits, dim=0)
            alpha = 1.0
            F = 1.0
            return Y_logits, Y_hat, alpha, F, alpha

        F = F.squeeze()
        Y_logits = torch.matmul(F, self.linear)  # [ks]
        Y_hat = torch.argmax(Y_logits, dim=0)

        return Y_logits, Y_hat, alpha, F, alpha

    def calculate_objective(self, X, Y, epoch):
        Y0 = Y.squeeze().long()
        target = torch.zeros(2).to(X.device)
        target[Y0] = 1

        Y_logits, Y_hat, weights, feature, weight = self.forward(X, Y0, epoch)

        loss = torch.nn.CrossEntropyLoss()
        all_loss = loss(Y_logits, target)

        return all_loss, Y_hat, weight


if __name__ == "__main__":
    x = torch.randn(64, 167).cuda()
    m = MILModel(device="cuda", dataset="MUSK1").cuda()
    for p, t in m.named_parameters():
        print(p)
    y = torch.Tensor([1]).squeeze().long().cuda()

    m.calculate_objective(x, y)
