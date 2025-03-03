import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
from sklearn.metrics import f1_score
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
from model.PSMIL import PSMIL


class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]
        img_pos = np.asarray(
            [int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])])  # row, col
        sample = {'input': img, 'position': img_pos}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                     transform=Compose([
                                         ToTensor()
                                     ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def test(args, bags_list, milnet):
    milnet.eval()
    num_bags = len(bags_list)

    psmil = PSMIL("TCGA", "no", False, 1).cuda()

    state_dict_weights = torch.load(os.path.join('weights', '20241126', args.weights[0] + ".pth"))
    psmil.load_state_dict(state_dict_weights, strict=True)
    for i in range(0, num_bags):
        feats_list = []
        pos_list = []
        classes_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            # bag_prediction, A, _ = milnet.b_classifier(bag_feats, ins_classes)
            # bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            bag_prediction, _, _, ins_prediction, _ = psmil(bag_feats, torch.tensor(-1), bag_feats.shape[0], "psa",
                                                            "TCGA")
            # print(bag_prediction,ins_prediction)
            bag_prediction = torch.argmax(bag_prediction, 1).cpu().numpy()
            ins_prediction = torch.argmax(ins_prediction, 1).cpu().numpy()
            color = [0, 0, 0]
            if bag_prediction == 1:
                print(bags_list[i] + ' is detected as malignant')
                color = [1, 0, 0]
                attentions = ins_prediction
            else:
                attentions = ins_prediction
                print(bags_list[i] + ' is detected as benign')
            color_map = np.zeros((np.amax(pos_arr, 0)[0] + 1, np.amax(pos_arr, 0)[1] + 1, 3))
            # attentions = attentions.cpu().numpy()
            # attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
            for k, pos in enumerate(pos_arr):
                tile_color = np.asarray(color) * attentions[k]
                color_map[pos[0], pos[1]] = tile_color
            slide_name = bags_list[i].split(os.sep)[-1]
            color_map = transform.resize(color_map, (color_map.shape[0] * 32, color_map.shape[1] * 32), order=0)
            io.imsave(os.path.join('test-c16', 'output', slide_name + '.png'), img_as_ubyte(color_map),
                      check_contrast=False)

    fold_predictions = []
    for fold_weight in args.weights:
        psmil = PSMIL("TCGA", "no", False, 1).cuda()

        state_dict_weights = torch.load(os.path.join('weights', '20241126', fold_weight + ".pth"))
        psmil.load_state_dict(state_dict_weights, strict=True)
        best_model = psmil
        criterion = torch.nn.NLLLoss()
        reserved_testing_bags = glob.glob('temp_train/test*.pt')
        test_loss_bag, avg_score, aucs, thresholds_optimal, test_predictions, test_labels = test_metrics(args,
                                                                                                         reserved_testing_bags,
                                                                                                         best_model.cuda(),
                                                                                                         criterion,
                                                                                                         thresholds=None,
                                                                                                         return_predictions=True)
        fold_predictions.append(test_predictions)
    predictions_stack = np.stack(fold_predictions, axis=0)
    mode_result = mode(predictions_stack, axis=0)
    combined_predictions = mode_result.mode[0]
    combined_predictions = combined_predictions.squeeze()

    # Compute Hamming Loss
    hammingloss = hamming_loss(test_labels, combined_predictions)
    print("Hamming Loss:", hammingloss)
    # Compute Subset Accuracy
    subset_accuracy = accuracy_score(test_labels, combined_predictions)
    labels = np.argmax(test_labels, axis=1)
    predictions = np.argmax(combined_predictions, axis=1)
    f1 = f1_score(labels, predictions, average="weighted")
    print("Test Accuracy (Exact Match Ratio):", subset_accuracy, f1)


def test_metrics(args, test_df, milnet, criterion, thresholds=None, return_predictions=False):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, item in enumerate(test_df):
            stacked_data = torch.load(item, map_location='cuda:0')
            bag_label = Tensor(stacked_data[0, args.feats_size:]).unsqueeze(0)
            bag_feats = Tensor(stacked_data[:, :args.feats_size])
            bag_feats = bag_feats.view(-1, args.feats_size)
            bag_prediction, _, _, ins_prediction, _ = milnet(bag_feats, bag_label, bag_feats.shape[0], "psa",
                                                             "TCGA")

            bag_loss = criterion(bag_prediction.view(1, -1), torch.argmax(bag_label, dim=1))
            loss = bag_loss

            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            test_predictions.extend([torch.exp(bag_prediction).squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)

    for i in range(args.num_classes):
        class_prediction_bag = copy.deepcopy(test_predictions[:, i])
        class_prediction_bag[test_predictions[:, i] >= 0.5] = 1
        class_prediction_bag[test_predictions[:, i] < 0.5] = 0
        test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)

    if return_predictions:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        # c_auc = roc_auc_score(label, prediction)
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print("ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
        # print(aucs)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_tumor', type=float, default=0.5282700061798096)
    parser.add_argument('--weights', type=str, nargs='+',
                        default=["fold_0_13", "fold_1_13", "fold_2_13", "fold_3_13", "fold_4_13"],
                        help='List of weights files to use')
    args = parser.parse_args()

    resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=1).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=1).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    trained_dict = torch.load(r'weights\20241120\fold_0_11.pth')
    milnet.load_state_dict(trained_dict, strict=False)
    aggregator_w = trained_dict["i_classifier.fc.0.weight"]
    aggregator_b = trained_dict["i_classifier.fc.0.bias"]
    # aggregator_weights = torch.load('example_aggregator_weights/c16_aggregator.pth')
    # milnet.load_state_dict(aggregator_weights, strict=False)

    state_dict_weights = torch.load(os.path.join('test-c16', 'weights', 'embedder0.pth'))
    # for i,(k,v) in enumerate(state_dict_weights.items()):
    #     print(k)
    new_state_dict = OrderedDict()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=1).cuda()
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = i_classifier.state_dict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v

    # new_state_dict["fc.weight"] = aggregator_weights["i_classifier.fc.0.weight"]
    # new_state_dict["fc.bias"] = aggregator_weights["i_classifier.fc.0.bias"]
    new_state_dict["fc.weight"] = aggregator_w
    new_state_dict["fc.bias"] = aggregator_b
    i_classifier.load_state_dict(new_state_dict, strict=True)
    milnet.i_classifier = i_classifier

    bags_list = glob.glob(os.path.join(r'test-c16', 'patches', '*'))
    os.makedirs(os.path.join('test-c16', 'output'), exist_ok=True)
    test(args, bags_list, milnet)
