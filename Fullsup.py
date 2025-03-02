import torch
import torch.nn as nn
import torch.optim as optim
from augment.autoaugment_extra import SVHNPolicy
from augment.cutout import Cutout
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import transforms, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, \
    RandomErasing, \
    ToPILImage
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch IMIPL')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Choose dataset', choices=['CIFAR10', "CIFAR100"])
parser.add_argument('--model', type=str, default='resnet18', help='Choose dataset',
                    choices=['resnet101', "fc", "resnet18"])
parser.add_argument('--classifier', type=bool, default=False, help='Choose mode')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== Training Settings ===")
print(f"Epochs: {args.epochs}")
print(f"Learning Rate: {args.lr}")
print(f"Dataset: {args.dataset}")
print(f"Model: {args.model}")
print(f"OnlyClassifier: {args.classifier}")


def visualize_clusters(model, dataloader):
    model.eval()  # 切换到评估模式
    features = []
    labels_list = []
    model = nn.Sequential(*list(model.children())[:-1])

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)

            # 提取特征
            outputs = model(images)
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
    # 设置点的尺寸为之前的一半a
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=sampled_labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
    plt.title('CIFAR10 Features used by ResNet18(Sup.)')
    plt.show()
class SAMClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SAMClassifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)  # 假设特征矩阵的通道数为256

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 定义数据加载器
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform0 = transforms.Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform1 = transforms.Compose([
    ToTensor(),
    Cutout(n_holes=1, length=20),
    ToPILImage(),
    SVHNPolicy(),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform0_ = transforms.Compose([
    RandomHorizontalFlip(),
    RandomCrop(28, 4, padding_mode='reflect'),
    ToTensor(),
    Normalize(mean=[0.1307], std=[0.3081]),
])

transform1_ = transforms.Compose([
    RandomHorizontalFlip(),
    RandomCrop(28, 4, padding_mode='reflect'),
    ToTensor(),
    Cutout(n_holes=1, length=16),
    ToPILImage(),
    ToTensor(),
    Normalize(mean=[0.1307], std=[0.3081]),
])

if args.dataset == "CIFAR10":
    trainset = datasets.CIFAR10(root='./datasets', train=True, download=False, transform=train_transform)

    testset = datasets.CIFAR10(root='./datasets', train=False, download=False, transform=test_transform)

    cls_num = 10
else:
    trainset = datasets.CIFAR100(root='./datasets', train=True, download=False, transform=train_transform)

    testset = datasets.CIFAR100(root='./datasets', train=False, download=False, transform=test_transform)

    cls_num = 100

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

if args.model == "deit":
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to(device)

    if args.classifier:
        for param in model.parameters():
            param.requires_grad = False

    # 替换分类层
    model.head = nn.Linear(model.head.in_features, cls_num).to(device)
elif args.model == "resnet101":
    model = models.resnet101(pretrained=True).to(device)

    if args.classifier:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, cls_num).to(device)
elif args.model == "resnet18":
    model = models.resnet18(pretrained=True).to(device)
else:
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    # if args.classifier:
    #     for param in model.parameters():
    #         param.requires_grad = False

if args.classifier:
    weights_path = f'.\weights\{args.model}_{args.dataset}_OnlyClassifier_fullsupweights.pth'  # 预训练权重文件的路径
else:
    weights_path = f'.\weights\{args.model}_{args.dataset}_TuneAll_fullsupweights.pth'

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


# 定义训练和测试函数
def train(epoch):
    model.train()
    pbar = tqdm(trainloader)



    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Train Epoch: {epoch} Loss: {loss.item():.6f}")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    return correct / len(testloader.dataset)



# 训练并打印测试准确率
accs = []

for epoch in range(1, args.epochs + 1):
    # visualize_clusters(model, testloader)
    train(epoch)
    acc = test()
    accs.append(acc)

# 保存模型权重
torch.save(model.state_dict(), weights_path)
