from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import torch.nn as nn
from model.WideResnet import WideResNet
from model.TransMIL import TransMIL
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import warnings
import sys
from torchvision import models
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 忽略ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)


def visualize_clusters(model, dataloader, name):
    model.eval()  # 切换到评估模式
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = []
    labels_list = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)

            # 提取特征
            _, _, _, _, outputs = model(images, 0, 64, "pool", "instance")
            features.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features, axis=0).squeeze()
    labels_list = np.concatenate(labels_list, axis=0)

    # 对数据进行采样，仅选取1000个点
    idx = np.random.choice(range(features.shape[0]), size=5000, replace=False)
    sampled_features = features[idx]
    sampled_labels = labels_list[idx]

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=0)
    # print(sampled_labels.shape,sampled_features.shape)
    reduced_features = tsne.fit_transform(sampled_features)

    # 绘制可视化图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=sampled_labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
    title = f'SVHN Features trained by {name}'
    plt.title(title)

    # 保存图像
    plt.savefig(f"{title}.png")
    # plt.show()


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


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


def adjust_cluster(alpha, n_cls):
    alpha = alpha.squeeze(0)

    alpha_clustered = alpha.cpu().detach().numpy().reshape(-1, 1)
    beta = torch.ones(alpha_clustered.shape[0]).to(alpha.device)

    if alpha_clustered.shape[0] > 1:
        # 初始化参数
        range_n_clusters = list(range(2, n_cls))  # 尝试的聚类数量范围
        silhouette_scores = []

        # 遍历不同的聚类数量
        for n_clusters in range_n_clusters:
            # 进行KMeans聚类
            clusterer = KMeans(n_clusters=n_clusters, n_init=5, random_state=10)
            cluster_labels = clusterer.fit_predict(alpha_clustered)

            # 计算Silhouette系数
            unique_labels = np.unique(cluster_labels)

            if len(unique_labels) > 1:
                silhouette_avg = silhouette_score(alpha_clustered, cluster_labels)
                silhouette_scores.append(silhouette_avg)

        if len(silhouette_scores) > 0 and max(silhouette_scores) > 0.9:
            # 找出最大Silhouette系数对应的聚类数量
            optimal_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]

            # 使用最佳聚类数量进行KMeans聚类
            clusterer = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=10)
            cluster_labels = clusterer.fit_predict(alpha_clustered)

            # 计算每个类簇的平均值并找到最大均值
            cluster_means = [alpha_clustered[cluster_labels == i].mean() for i in range(optimal_clusters)]
            min_mean_cluster = cluster_means.index(min(cluster_means))
            # 找到属于最大均值类簇的元素在alpha中的下标列表
            cluster_indices = np.where(cluster_labels == min_mean_cluster)[0]
            for i in range(optimal_clusters):
                if i == min_mean_cluster:
                    beta[cluster_labels == i] = 0.9

    # 如果不满足条件，返回alpha的全部下标
    return beta  # list(range(len(alpha)))


class StretchSig(nn.Module):
    def __init__(self, lam):
        super(StretchSig, self).__init__()
        self.lam = lam

    def forward(self, input):
        output = self.lam * torch.sigmoid(input / self.lam)
        output = output / torch.sum(output)
        return output


def calculate_weights(prob_matrix):
    # 利用概率矩阵计算熵。这里假设概率矩阵的每一行都是一个概率分布。
    epsilon = 1e-10  # 避免对0取对数
    log_probs = torch.log(prob_matrix + epsilon)
    entropy = torch.sum(prob_matrix * log_probs, dim=1)  # 按行求和计算负熵
    weights = F.softmax(entropy, dim=0)
    return weights.unsqueeze(0)


def ent_loss(alpha, indices):
    # 选择alpha中特定的元素
    selected_alpha = alpha[indices]
    all_indices = set(range(len(alpha)))
    non_selected_indices = list(all_indices - set(indices))
    non_selected_alpha = alpha[non_selected_indices]

    # 计算这些元素的和
    sum_selected_alpha = torch.sum(selected_alpha)

    # (可选) 计算熵部分的损失，确保归一化（即所有元素和为1）
    entropy_loss = -torch.sum(non_selected_alpha * torch.log(non_selected_alpha + 1e-9))  # 加上一个小常数以避免对0取对数
    entropy_loss += -sum_selected_alpha * torch.log(sum_selected_alpha)

    return entropy_loss


class Deit(nn.Module):
    def __init__(self):
        super(Deit, self).__init__()
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.fc = nn.Linear(1000, 512)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


class Backbone(nn.Module):
    def __init__(self, dataset, backbone):
        super(Backbone, self).__init__()

        self.stem = None

        if backbone == "resnet18":
            self.stem = models.resnet18(pretrained=True)
            self.stem.fc = nn.Identity()
            self.feat_dim = 512
        elif backbone == "resnet101":
            self.stem = models.resnet101(pretrained=True)
            self.stem.fc = nn.Identity()
            self.feat_dim = 2048
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Flatten(),
                nn.Linear(9216, 512),
                nn.ReLU(),
            )
            # freeze self.stem
            self.feat_dim = 512

    def forward(self, x):
        x = self.stem(x)
        return x.view(x.shape[0], -1)  # [t,o]


class PSAMIL(nn.Module):
    def __init__(self, dataset="CIFAR10", backbone="deit", need_cl=False, pos_cls=1):
        super(PSAMIL, self).__init__()

        self.backbone = backbone
        self.stem = Backbone(dataset, backbone)
        for param in self.stem.parameters():
            param.requires_grad = False
        self.dataset = dataset
        self.need_cl = need_cl
        self.ks = 0

        if dataset == "CIFAR10":
            self.ks = 10 - pos_cls + 1
        elif dataset == "CIFAR100":
            self.ks = 100 - pos_cls + 1
        elif dataset == "SVHN":
            self.ks = 10 - pos_cls + 1
        elif dataset == "FMNIST":
            self.ks = 10 - pos_cls + 1
        elif dataset == "UNBC":
            self.ks = 4 - pos_cls + 1

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

        # used for transmil
        self.tpt = TransMIL(n_classes=self.ks)

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
        if not self.need_cl:  # no cl
            # x:[t,c,h,w]
            # print(x)
            fs = self.stem(x)  # [t,o]
            ins_probs = torch.softmax(torch.mm(fs, self.linear), dim=1)  # [t,ks]
            if testmode == "instance":
                features = fs  # [1,o]
                Y_logit = torch.matmul(fs, self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]
                Y_hat = torch.argmax(Y_prob, dim=1).squeeze()  # scalar

                return Y_prob, Y_hat.unsqueeze(0), Y_hat, Y_hat, features  # [1,ks],[1], [1], [1], [1,o]
            if self.firsttime:
                self.fbank.data = fs.mean(dim=0, keepdim=True).repeat(self.ks, 1).transpose(0, 1)
                self.firsttime = False

            if pooling == "psa":
                with torch.no_grad():
                    pred = torch.softmax(torch.mm(fs, self.fbank), dim=1)  # [t,ks]
                    predicted_labels = torch.argmax(pred, dim=1)  # Find the predicted class for each sample
                    pred_one_hot = nn.functional.one_hot(predicted_labels, num_classes=pred.shape[1]).float()

                alpha = self.ps_attention(pred)  # [t,1]
                alpha1 = self.ps_attention(pred_one_hot)  # [t,1]
                alpha = torch.softmax(alpha.T, dim=1)  # [1,t]
                alpha1 = torch.softmax(alpha1.T, dim=1)  # [1,t]

                alpha = (alpha + alpha1) / 2

                F = torch.mm(alpha, fs)  # [1,o]
                selected_samples = (predicted_labels == y)

                if torch.any(selected_samples):
                    selected_indices = torch.nonzero(selected_samples).squeeze()  # 获取为True的索引
                    critical_fs = fs[selected_indices]  # 获取关键特征
                    criticalF = torch.mean(critical_fs, dim=0, keepdim=True)
                else:
                    criticalF = F

                if y >= 0:
                    newf = nn.functional.normalize((0.999 * self.fbank[:, y]) + (0.001 * criticalF.t()), dim=0)
                    self.fbank[:, y] = newf

                if torch.isnan(self.fbank).any():
                    print("Tensor Fbank contains NaN values")
                    sys.exit()
                Y_logit = torch.matmul(F, self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]

            if pooling == "fsa":
                alpha = self.attention_based(fs)  # [t,1]
                alpha = torch.softmax(alpha.T, dim=1)  # [1,t]

                F = torch.mm(alpha, fs)  # [1,o]

                Y_logit = torch.matmul(F, self.linear)  # [1,ks]

                if torch.isnan(self.fbank).any():
                    print("Tensor Fbank contains NaN values")
                    sys.exit()
                Y_prob = self.output(Y_logit)  # [1,ks]

            if pooling == "mha":
                A_V = self.att_layer_V(fs, self.linear_V.weight, self.linear_V.bias)
                A_U = self.att_layer_U(fs, self.linear_U.weight, self.linear_U.bias)
                A = self.attention_weights(A_V * A_U)
                # alpha = torch.softmax(A.T, dim=1)  # [1,t]
                alpha = torch.transpose(A, 1, 0)
                alpha = torch.sigmoid(alpha)
                c = torch.sum(alpha)
                alpha = alpha / c
                F = torch.mm(alpha, fs)  # Equation (3)
                Y_logit = torch.matmul(F, self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]

            if pooling == "dta":
                slide_pseudo_feat = []
                slide_sub_preds = []
                if testmode == "bag":
                    inputs_pseudo_bags = torch.chunk(fs.squeeze(0), 10, dim=0)
                else:
                    inputs_pseudo_bags = torch.chunk(fs, 1, dim=0)

                for subFeat_tensor in inputs_pseudo_bags:
                    # slide_sub_labels.append(labels)
                    tmidFeat = subFeat_tensor.to(fs.device)
                    tAA = self.dta(tmidFeat).squeeze(0)
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                    slide_pseudo_feat.append(tattFeat_tensor)
                    tPredict = torch.softmax(torch.matmul(tattFeat_tensor, self.linear), dim=1)  # [1,ks]
                    slide_sub_preds.append(tPredict)

                ins_probs = torch.cat(slide_sub_preds, dim=0)  # numGroup x ks
                slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
                Y_logit = self.dtac(slide_pseudo_feat)
                Y_prob = self.output(Y_logit)  # [1,ks]
                alpha = 0.0

            # used for tsp
            if pooling == "tpt":
                # used for tsp
                Y_logit = self.tpt(fs.unsqueeze(0))  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]
                alpha = 0.0

            if pooling == "dsa":
                if testmode == "bag":
                    V = fs  # default no change [t,o]
                    Q = self.q(fs).view(x.shape[0], -1)  # [t,q], unsorted
                    c = ins_probs  # [t,ks],
                    idx = torch.argmax(c[:, y])  # [1]
                    m_feat = fs[idx].unsqueeze(0)  # select critical instances, m_feats in shape [1,o]

                    q_max = self.q(m_feat)  # compute queries of critical instances, q_max in shape [1,q]
                    A = torch.matmul(Q,
                                     q_max.T)  # inner product A in shape [ks,t], contains unnormalized attention scores

                    alpha = nn.functional.softmax(A, dim=0).T  # normalize attention scores, A in shape [ks,t]

                    F = torch.mm(alpha, V)  # compute bag representation, B in shape [ks,o]
                    F = (F + fs[idx]) / 2
                    Y_logit = torch.matmul(fs[idx].unsqueeze(0), self.linear)  # [1,ks]
                    Y_prob = self.output(Y_logit)  # [1,ks]
                    ins_probs = Y_prob
                else:
                    alpha = 0.
                    F = fs
                Y_logit = torch.matmul(F, self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]

            if pooling == "avg":
                Y_logit = torch.matmul(torch.mean(fs, dim=0, keepdim=True), self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]
                alpha = 0.0

            Y_hat = torch.argmax(Y_prob, dim=1).squeeze()  # scalar
            return Y_prob, Y_hat.unsqueeze(0), alpha, ins_probs, F  # [1,ks],[1], [1,t], [t,ks]
        else:  # with cl term
            # x:[Nt,c,h,w]
            fs = self.stem(x)  # [Nt,o]
            N = fs.shape[0] // bag_size

            if N <= 0:
                N = 1
            if not testmode == "instance":
                features = fs.view(N, bag_size, -1)  # [N,t,o]
                fs = torch.mean(features, dim=0)  # [t,o]
            else:
                features = fs  # [1,o]
                Y_logit = torch.matmul(fs, self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]
                Y_hat = torch.argmax(Y_prob, dim=1).squeeze()  # scalar

                return Y_prob, Y_hat.unsqueeze(0), Y_hat, Y_hat, features  # [1,ks],[1], [1], [1], [1,o]
            with torch.no_grad():
                ps = torch.softmax(torch.matmul(fs, self.linear), dim=-1)  # [t,ks]
                ins_probs = ps

            if self.firsttime:
                self.fbank.data = fs.mean(dim=0, keepdim=True).repeat(self.ks, 1).transpose(0, 1)
                self.firsttime = False

            if pooling == "psa":
                with torch.no_grad():
                    pred = torch.softmax(torch.einsum("nto,ok->ntk", features, self.fbank), dim=-1)  # [N,t,ks]
                    predicted_labels = torch.argmax(pred[0], dim=-1)  # Find the predicted class for each sample
                    pred_one_hot = nn.functional.one_hot(predicted_labels, num_classes=self.ks).float()

                alpha = self.ps_attention(pred)  # [N,t,1]
                alpha1 = self.ps_attention(pred_one_hot)  # [t,1]
                # alpha = torch.transpose(alpha[0], 1, 0)
                # alpha = torch.sigmoid(alpha)
                # alpha = alpha/torch.sum(alpha)
                # alpha1 = torch.transpose(alpha1, 1, 0)
                # alpha1 = torch.sigmoid(alpha1)
                # alpha1 = alpha/torch.sum(alpha1)
                alpha = torch.softmax(alpha[0].T, dim=1)  # [1,t]
                alpha1 = torch.softmax(alpha1.T, dim=1)  # [1,t]

                alpha = (alpha + alpha1) / 2

                F = torch.mm(alpha, features[0])  # [1,o]

                selected_samples = (predicted_labels == y)  # Compare with the target class y

                if any(selected_samples):
                    critical_fs = features[0][selected_samples]  # [n,o]
                    criticalF = torch.mean(critical_fs, dim=0, keepdim=True)
                else:
                    criticalF = F

                if y >= 0:
                    newf = nn.functional.normalize((0.999 * self.fbank[:, y]) + (0.001 * criticalF.t()), dim=0)
                    self.fbank[:, y] = newf

                if torch.isnan(self.fbank).any():
                    print("Tensor Fbank contains NaN values")
                    sys.exit()
                pred = [torch.softmax(torch.einsum("nto,ok->ntk", features, self.linear), dim=-1)]
                Y_logit = torch.matmul(F, self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]

            if pooling == "fsa":
                alpha = self.attention_based(features[0])  # [t,1]
                alpha = torch.softmax(alpha.T, dim=1)  # [1,t]
                F = torch.mm(alpha, features[0])  # [1,o]
                Y_logit = torch.matmul(F, self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]
                pred = [torch.softmax(torch.einsum("nto,ok->ntk", features, self.linear), dim=-1)]  # [N,t,ks]

            if pooling == "mha":
                A_V = self.att_layer_V(features[0], self.linear_V.weight, self.linear_V.bias)
                A_U = self.att_layer_U(features[0], self.linear_U.weight, self.linear_U.bias)
                A = self.attention_weights(A_V * A_U)
                # alpha = torch.softmax(A.T, dim=1)  # [1,t]
                alpha = torch.transpose(A, 1, 0)
                alpha = torch.sigmoid(alpha)
                with torch.no_grad():
                    c = torch.sum(alpha)
                alpha = alpha / c
                F = torch.mm(alpha, features[0])  # Equation (3)
                Y_logit = torch.matmul(F, self.linear)  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]
                pred = [torch.softmax(torch.einsum("nto,ok->ntk", features, self.linear), dim=-1)]  # [N,t,ks]

            # used for tsp
            if pooling == "tpt":
                # used for tsp
                Y_logit = self.tpt(features[0].unsqueeze(0))  # [1,ks]
                Y_prob = self.output(Y_logit)  # [1,ks]
                pred = [torch.softmax(torch.einsum("nto,ok->ntk", features, self.linear), dim=-1)]  # [N,t,ks]
                alpha = 0.0

            if pooling == "dta":
                slide_pseudo_feat = []
                slide_sub_preds = []
                if testmode == "bag":
                    inputs_pseudo_bags = torch.chunk(features[0].squeeze(0), 10, dim=0)
                else:
                    inputs_pseudo_bags = torch.chunk(features[0], 1, dim=0)

                for subFeat_tensor in inputs_pseudo_bags:
                    # slide_sub_labels.append(labels)
                    tmidFeat = subFeat_tensor.to(fs.device)
                    tAA = self.dta(tmidFeat).squeeze(0)
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                    slide_pseudo_feat.append(tattFeat_tensor)
                    tPredict = torch.softmax(torch.matmul(tattFeat_tensor, self.linear), dim=1)  # [1,ks]
                    slide_sub_preds.append(tPredict)

                ins_probs = torch.cat(slide_sub_preds, dim=0)  # numGroup x ks
                pred = [torch.softmax(torch.einsum("nto,ok->ntk", features, self.fbank), dim=-1), ins_probs]  # [N,t,ks]
                slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
                Y_logit = self.dtac(slide_pseudo_feat)
                Y_prob = self.output(Y_logit)  # [1,ks]
                alpha = 0.0

            Y_hat = torch.argmax(Y_prob, dim=1).squeeze()  # scalar
            return Y_prob, Y_hat.unsqueeze(0), alpha, pred, F  # [1,ks],[1], [1,t], [t,ks]


if __name__ == "__main__":
    x = torch.randn(10, 3, 32, 32).cuda()
    m = PSAMIL(dataset="CIFAR10", pos_cls=5).cuda()
    y = torch.tensor(2).cuda()

    m.calculate_objective(x, y, "rga", 1)
