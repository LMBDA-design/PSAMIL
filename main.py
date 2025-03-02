import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets.MILDataset import MILDataset, UNBCBagDataset, get_sampler
from datasets.svhn import MY_SVHN
from datasets.fmnist import MY_FMNIST
import torch.utils.data as data_utils
import torchvision.datasets as datasets
from tqdm import tqdm
from model.PSAMIL import PSAMIL, visualize_clusters
from torchvision import transforms as T
import os
import numpy as np
from util import ICC, PCC, MAE, MSE
import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch PSAMIL')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--pos_cls', type=int, default=1, metavar='T',
                    help='positive label start from')
parser.add_argument('--bag_size', type=int, default=64, metavar='BL',
                    help='bag size')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Choose dataset',
                    choices=['CIFAR10', "UNBC", "CIFAR100"])
parser.add_argument('--backbone', type=str, default='resnet18', help='Choose dataset',
                    choices=['resnet18', "deit", "resnet101"])
parser.add_argument('--pooling', type=str, default='psa', help='Choose pooling component',
                    choices=['psa', 'fsa', ])
parser.add_argument('--finetune', type=bool, default=True, help='Whether backbone being trained')
parser.add_argument('--cl', type=bool, default=True, help='Whether use cl term')
parser.add_argument('--start', type=int, default=0, help='Start Epoch')

args = parser.parse_args()

print("=== Training Settings ===")
print(f"Epochs: {args.epochs}")
print(f"Learning Rate: {args.lr}")
print(f"Positive Label Start From: {args.pos_cls}")
print(f"Bag Size: {args.bag_size}")
print(f"Dataset: {args.dataset}")
print(f"Backbone: {args.backbone}")
print(f"Pooling Component: {args.pooling}")
print(f"If no finetune,no backbone trained,no cl.Fine-tune: {args.finetune}")
print(f"Contain Contrastive Term:  {args.cl}")

# 检查是否有可用的GPU，如果有的话使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OnlyClassifier_weights_path = f'.\milweights\{args.backbone}_{args.dataset}_OnlyClassifier_{args.pooling}_{args.epochs}_fbank.pth'  # 预训练权重文件的路径
Finetune_weights_path = f'.\milweights\{args.backbone}_{args.dataset}_FineTune_{args.pooling}_{args.epochs}_{args.cl}.pth'  # 预训练权重文件的路径

if args.dataset == "CIFAR10":
    if not args.finetune:
        train_set = MILDataset(datasetname=args.dataset, bag_size=args.bag_size, pos_cls=args.pos_cls,
                               need_cl=False)
        model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=False, pos_cls=args.pos_cls).to(device)
    else:
        train_set = MILDataset(datasetname=args.dataset, bag_size=args.bag_size, pos_cls=args.pos_cls,
                               need_cl=args.cl)
        model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=args.cl, pos_cls=args.pos_cls).to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    instance_test_set = torchvision.datasets.CIFAR10(root="datasets", train=False, download=False,
                                                     transform=train_set.transform0)
elif args.dataset == "CIFAR100":
    if not args.finetune:
        train_set = MILDataset(datasetname=args.dataset, bag_size=args.bag_size, pos_cls=args.pos_cls,
                               need_cl=False)
        model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=False, pos_cls=args.pos_cls).to(device)
    else:
        train_set = MILDataset(datasetname=args.dataset, bag_size=args.bag_size, pos_cls=args.pos_cls,
                               need_cl=args.cl)
        model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=args.cl, pos_cls=args.pos_cls).to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    instance_test_set = torchvision.datasets.CIFAR100(root="datasets", train=False, download=True,
                                                      transform=train_set.transform0)
elif args.dataset == "SVHN":
    if not args.finetune:
        train_set = MILDataset(datasetname=args.dataset, bag_size=args.bag_size, pos_cls=args.pos_cls,
                               need_cl=False)
        model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=False, pos_cls=args.pos_cls).to(device)
    else:
        train_set = MILDataset(datasetname=args.dataset, bag_size=args.bag_size, pos_cls=args.pos_cls,
                               need_cl=args.cl)
        model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=args.cl, pos_cls=args.pos_cls).to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    instance_test_set = MY_SVHN(root="datasets", split='test', download=False)
elif args.dataset == "FMNIST":
    if not args.finetune:
        train_set = MILDataset(datasetname=args.dataset, bag_size=args.bag_size, pos_cls=args.pos_cls,
                               need_cl=False)
        model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=False, pos_cls=args.pos_cls).to(device)
    else:
        train_set = MILDataset(datasetname=args.dataset, bag_size=args.bag_size, pos_cls=args.pos_cls,
                               need_cl=args.cl)
        model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=args.cl, pos_cls=args.pos_cls).to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    instance_test_set = MY_FMNIST(root="datasets", train=False, download=False)
else:
    target = 0
    train_set = UNBCBagDataset(target_number=target, mean_bag_length=args.bag_size, train=True)
    model = PSAMIL(dataset=args.dataset, backbone=args.backbone, need_cl=args.cl, pos_cls=args.pos_cls).to(device)
    sampler = get_sampler(train_set)
    train_loader = data_utils.DataLoader(
        train_set, batch_size=1, )
    # sampler=sampler)
    instance_test_set = UNBCBagDataset(target_number=target, train=False, mode="single")

test_loader = torch.utils.data.DataLoader(instance_test_set, batch_size=1, shuffle=False)

# ins_labels_est = torch.ones([len(train_set.dataset)], device=device).long()
# ins_confidence = torch.ones_like(ins_labels_est, device=device).float()

if args.start > 0:
    if os.path.exists(Finetune_weights_path):
        pretrained_weights = torch.load(Finetune_weights_path)
        model.load_state_dict(pretrained_weights)
        print("load success!")

# Define the loss function and optimizer
bag_criterion = torch.nn.NLLLoss()
# bag_criterion = torch.nn.NLLLoss(weight=torch.Tensor([1,2,2,10]).to(device))
ins_loss = torch.nn.CrossEntropyLoss()
bag_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)




# Define the training function
def train(epoch):
    model.train()
    correct = 1
    total = 0
    entropy = 0
    w_critical_total = 0
    stddev = 0
    # lam = epoch * 0.01 + 0.01 * args.start
    # lam = 0.01  * (args.start+args.epochs)
    # if lam > 0.1:
    #     lam = 0.1
    lam = 0

    print("epoch,lam = ", args.start, epoch, lam)

    progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

    if args.finetune:
        if not args.cl:  # normal mode,tends to fail easily
            for batch_idx, (bagImages0, bag_label, ins_labels, ins_indices) in enumerate(progress_bar):
                bagImages0, bag_label, ins_labels, ins_indices = bagImages0.squeeze(0).to(
                    device), bag_label.long().to(
                    device), \
                    ins_labels.squeeze(0).to(device), torch.tensor(ins_indices).to(
                    device)  # Move inputs and targets to GPU
                bag_optimizer.zero_grad()

                Y_prob, Y_hat, weights, ins_probs, _ = model(bagImages0, bag_label, args.bag_size, args.pooling, "bag")

                bag_loss = bag_criterion(Y_prob, bag_label)
                correct += bag_loss.cpu().item()
                loss = bag_loss

 
                idx = (ins_labels >= args.pos_cls)
                if bag_label == 0:
                    w_critical_total += 1.
                else:
                    w_critical_total += torch.sum(weights[0][idx]).detach().cpu().item()
                    pos_weights = weights[0][idx]
                    if torch.all(torch.eq(pos_weights, pos_weights[0])):
                        # 如果全相等,直接返回0
                        stddev += 0.
                    else:
                        stddev += torch.std(pos_weights).detach().cpu().item()
                total += 1

                entropy_per_sample = -torch.sum(ins_probs * torch.log(ins_probs + 1e-9), dim=1)
                # 计算熵的均值
                mean_entropy = torch.mean(entropy_per_sample)
                entropy += mean_entropy

                loss.backward()
                bag_optimizer.step()

                progress_bar.set_postfix(
                    {"Training loss": correct / total, "critical_weights": w_critical_total / total,
                     "std": stddev / total, "entropy": entropy.cpu().item() / total})
        else:  # finetune with cl term
            if args.start > 0 or epoch > 0:
                for param in model.stem.parameters():
                    param.requires_grad = True
            else:
                for param in model.stem.parameters():
                    param.requires_grad = False

            for batch_idx, (
                    bagImages0, bagImages1, bagImages2, bag_label, ins_labels, ins_indices) in enumerate(progress_bar):

                bagImages0, bagImages1, bagImages2, bag_label, ins_labels, ins_indices = bagImages0.squeeze(
                    0).to(
                    device), \
                    bagImages1.squeeze(0).to(device), \
                    bagImages2.squeeze(0).to(device), bag_label.long().to(device), \
                    ins_labels.squeeze(0).to(device), torch.tensor(ins_indices).to(
                    device)  # Move inputs and targets to GPU
                # print(batch_idx)
                bagImagesAll = torch.cat((bagImages0, bagImages1, bagImages2), dim=0)

                Y_prob, Y_hat, weights, preds, _ = model(bagImagesAll, bag_label, args.bag_size, args.pooling, "bag")


                idx = (ins_labels >= args.pos_cls)
                if bag_label == 0:
                    w_critical_total += 1.
                else:
                    w_critical_total += torch.sum(weights[0][idx]).detach().cpu().item()
                    pos_weights = weights[0][idx]
                    if torch.all(torch.eq(pos_weights, pos_weights[0])):
                        # 如果全相等,直接返回0
                        stddev += 0.
                    else:
                        stddev += torch.std(pos_weights).detach().cpu().item()
                total += 1

                # 计算几何平均概率
                preds, group_preds = preds[0], preds[-1]
                # 计算每个样本的熵
                entropy_per_sample = -torch.sum(preds * torch.log(preds + 1e-9), dim=1)
                # 计算熵的均值
                mean_entropy = torch.mean(entropy_per_sample)
                entropy += mean_entropy

                log_preds = torch.log(preds + 1e-9)  # 取对数
                mean_log_preds = torch.mean(log_preds, dim=0)  # 计算对数的平均值
                geom_mean_probs = torch.exp(mean_log_preds)  # 取指数得到几何平均数
                # 归一化几何平均概率
                estimated_probs = geom_mean_probs / (torch.sum(geom_mean_probs, dim=1, keepdim=True) + 1e-9)
                estimated_labels = torch.argmax(estimated_probs, dim=1)

                total_loss = 0

                # 对 preds 中的每一个 view 进行处理
                for i in range(preds.shape[0]):
                    # 计算每一个 view 的损失
                    loss = ins_loss(preds[i], estimated_labels)
                    # 将每一个 view 的损失加到总损失上
                    total_loss += loss

                average_loss = total_loss / preds.shape[0]
                # print(Y_prob.shape,bag_label.shape)
                bag_loss = bag_criterion(Y_prob, bag_label) + (lam * average_loss)

                if args.pooling == "dta":
                    labels = bag_label.repeat(group_preds.shape[0])

                    total_loss = 0

                    # 对 preds 中的每一个 subgroup 进行处理
                    for i in range(group_preds.shape[0]):
                        # 计算每一个 view 的损失
                        loss = ins_loss(group_preds[i], labels[i])
                        # 将每一个 view 的损失加到总损失上
                        total_loss += loss

                    average_loss = total_loss / group_preds.shape[0]
                    bag_loss = bag_loss + average_loss

                correct += bag_loss.cpu().item()

                # Backward pass and optimization
                bag_optimizer.zero_grad()

                bag_loss.backward()

                bag_optimizer.step()

                progress_bar.set_postfix(
                    {"Training loss": correct / total, "critical_weights": w_critical_total / total,
                     "std": stddev / total, "entropy": entropy.cpu().item() / total})

    else:  # onlyclassifier 
        for param in model.stem.parameters():
            param.requires_grad = False

        for batch_idx, (bagImages0, bag_label, ins_labels, ins_indices) in enumerate(progress_bar):
            bagImages0, bag_label, ins_labels, ins_indices = bagImages0.squeeze(0).to(
                device), bag_label.long().to(
                device), \
                ins_labels.squeeze(0).to(device), torch.tensor(ins_indices).to(device)  # Move inputs and targets to GPU
            bag_optimizer.zero_grad()

            Y_prob, Y_hat, weights, preds, _ = model(bagImages0, bag_label, args.bag_size, args.pooling, "bag")

            bag_loss = bag_criterion(Y_prob, bag_label)
            correct += bag_loss.cpu().item()
            loss = bag_loss

            idx = (ins_labels >= args.pos_cls)
            if bag_label == 0:
                w_critical_total += 1.
            else:
                w_critical_total += torch.sum(weights[0][idx]).detach().cpu().item()
                pos_weights = weights[0][idx]
                if torch.all(torch.eq(pos_weights, pos_weights[0])):
                    # 如果全相等,直接返回0
                    stddev += 0.
                else:
                    stddev += torch.std(pos_weights).detach().cpu().item()
            total += 1

            if args.pooling == "dta":
                # 计算几何平均概率
                labels = bag_label.repeat(preds.shape[0])

                total_loss = 0

                # 对 preds 中的每一个 subgroup 进行处理
                for i in range(preds.shape[0]):
                    # 计算每一个 view 的损失
                    loss = ins_loss(preds[i], labels[i])
                    # 将每一个 view 的损失加到总损失上
                    total_loss += loss

                average_loss = total_loss / preds.shape[0]
                loss = bag_criterion(Y_prob, bag_label) + average_loss
                # print(average_loss,loss)
                correct += loss.cpu().item()

            # if args.pooling == "dsa":
            #
            #     loss = bag_criterion(Y_prob, bag_label) + bag_criterion(preds, bag_label)
            #     # print(average_loss,loss)
            #     correct += loss.cpu().item()

            loss.backward()
            bag_optimizer.step()

            progress_bar.set_postfix(
                {"Training loss": correct / total, "critical_weights": w_critical_total / total, "std": stddev / total})

    return w_critical_total / total, stddev / total


# Define the testing function
def test(lam):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        if not args.dataset == "UNBC":
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.long().to(device)  # Move inputs and targets to GPU
                Y_probs, Y_hats, weights, ins_probs, _ = model(images, -1, args.bag_size, args.pooling, "instance")
                if Y_hats == labels:
                    correct += 1
                total += 1
        else:
            outs = []
            targets = []
            for batch_idx, (images, labels, bag_label, _) in enumerate(test_loader):
                images, bag_label = images.squeeze(0).to(device), bag_label.long().to(
                    device)  # Move inputs and targets to GPU
                bs, ncrops, c, h, w = images.shape
                predicted_bag_label = []
                for i in range(ncrops):
                    data_slice = images[:, i, :, :, :]
                    with torch.no_grad():
                        # print("slice",i)
                        bag_slice_predict, Y_hats, weights, ins_probs = model(data_slice, -1, args.bag_size,
                                                                              args.pooling, "instance")
                    predicted_bag_label.append(bag_slice_predict)
                    # (predicted_bag_label,bag_label)
                predicted_bag_label = torch.stack(predicted_bag_label).mean(0).squeeze(0)
                Y_hat = torch.argmax(predicted_bag_label, dim=0)
                if Y_hat == bag_label:
                    correct += 1
                total += 1
                targets.append(bag_label.cpu().squeeze().numpy())
                outs.append(Y_hat.cpu())
            outs = np.array(outs)
            targets = np.array(targets)
            outs = torch.from_numpy(outs)
            targets = torch.from_numpy(targets)
            icc = ICC(outs, targets)
            pcc = PCC(outs, targets)
            mae = MAE(outs, targets)
            mse = MSE(outs, targets)
            print(f"icc:{icc},pcc:{pcc},mae:{mae},mse:{mse}")

    print(f'Test Accuracy: {100 * correct / total}')
    return correct / total


# Train and test the model
accs = []
c_ws = []
c_devs = []
if args.start > 0:
    prefix = f"{args.pooling}_{args.dataset}_{args.start}"
    # find the .pth with prefix
    files = os.listdir(".")
    for file in files:
        if file.startswith(prefix) and file.endswith(".pth"):
            pth = file
            load_model(model, pth)
            break

for epoch in range(args.epochs):
    # acc = test(0)
    # visualize_clusters(model, test_loader, "PSMIL")
    # visualize_clusters(model, test_loader,"SVHN")
    c_w, c_dev = train(epoch)
    acc = test(0)
    accs.append(acc)
    c_ws.append(c_w)
    c_devs.append(c_dev)

print(accs, c_ws, c_devs)
save_model(model, f"{args.pooling}_{args.dataset}_{args.start + args.epochs}_{accs[-1]}.pth")
