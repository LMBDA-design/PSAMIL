from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from torchvision import models
class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls)

    def forward(self, x):  ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x)  ## K x L
        pred = self.classifier(afeat)  ## K x num_cls
        return pred
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

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

class Backbone(nn.Module):
    def __init__(self, dataset, backbone):
        super(Backbone, self).__init__()
        self.stem = nn.Identity()
        self.feat_dim = 512


    def forward(self, x):
        x = self.stem(x)
        return x.view(x.shape[0], -1)  # [t,o]


class PSMIL(nn.Module):
    def __init__(self, dataset="TCGA", backbone="resnet18", need_cl=False, pos_cls=1):
        super(PSMIL, self).__init__()

        self.backbone = backbone
        self.stem = Backbone(dataset, backbone)
        for param in self.stem.parameters():
            param.requires_grad = False
        self.dataset = dataset
        self.need_cl = need_cl
        self.ks = 2


        self.firsttime = True
        self.fbank = nn.Parameter(data=torch.FloatTensor(self.stem.feat_dim, self.ks), requires_grad=False)

        self.ps_attention = nn.Sequential(
            nn.Linear(self.ks, 1)
        )

        self.attention_based = nn.Sequential(
            nn.Linear(self.stem.feat_dim, 1)
        )

        self.att_layer_V = GatedAttentionLayerV(self.stem.feat_dim)
        self.att_layer_U = GatedAttentionLayerU(self.stem.feat_dim)
        self.linear_V = nn.Linear(self.stem.feat_dim, self.ks)
        self.linear_U = nn.Linear(self.stem.feat_dim, self.ks)
        self.attention_weights = nn.Sequential(
            nn.Linear(self.ks, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # used for dsp . q for query ,v for itself
        self.q = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # used for dtfdmil
        self.dta = Attention_Gated(L=self.stem.feat_dim, D=128, K=1)
        self.dtac = Attention_with_Classifier(L=self.stem.feat_dim, D=128, K=1, num_cls=self.ks)

        self.linear = nn.Parameter(data=torch.FloatTensor(self.stem.feat_dim, self.ks), requires_grad=True)
        nn.init.kaiming_uniform_(self.linear)

        self.output = nn.LogSoftmax(dim=1)  # [1,ks]

    def forward(self, x, y, bag_size, pooling, testmode="bag"):
        # x:[t,c,h,w]
        # print(x)
        fs = self.stem(x)  # [t,o]
        ins_probs = torch.softmax(torch.mm(fs, self.linear),dim=1)  # [t,ks]

        if self.firsttime:
            self.fbank.data = fs.mean(dim=0, keepdim=True).repeat(self.ks, 1).transpose(0, 1)
            self.firsttime = False

        with torch.no_grad():
            pred = torch.softmax(torch.mm(fs, self.fbank), dim=1)  # [t,ks]
            predicted_labels = torch.argmax(pred, dim=1)  # Find the predicted class for each sample


        alpha = self.ps_attention(pred)  # [t,1]

        alpha = torch.softmax(alpha.T, dim=1)  # [1,t]

        F = torch.mm(alpha, fs)  # [1,o]

        if torch.sum(y) >= 0:
            y = torch.argmax(y.squeeze())
            selected_samples = torch.argmax(pred[:, y])

            criticalF = fs[selected_samples]  # [o]
            newfy = nn.functional.normalize((0.99 * self.fbank[:, y]) + (0.01 * criticalF), dim=0)
            if torch.isnan(newfy).any():
                newfy = torch.mean(fs,dim=0)
            self.fbank[:, y] = newfy


            if torch.isnan(self.fbank).any():
                print("Tensor Fbank contains NaN values")
                sys.exit()

        Y_logit = torch.matmul(F, self.linear)  # [1,ks]
        Y_prob = self.output(Y_logit)  # [1,ks]


        Y_hat = torch.argmax(Y_prob, dim=1).squeeze()  # scalar
        return Y_prob, Y_hat.unsqueeze(0), alpha, ins_probs, F  # [1,ks],[1], [1,t], [t,ks]


if __name__ == "__main__":
    x = torch.randn(10, 3, 32, 32).cuda()
    m = PSMIL(dataset="CIFAR10", pos_cls=5).cuda()
    y = torch.tensor(2).cuda()

    m.calculate_objective(x, y, "rga", 1)
