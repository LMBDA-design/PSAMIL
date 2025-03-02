from model.PSAMIL import visualize_clusters,PSAMIL
import torch,torchvision
from torchvision.transforms import transforms, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, \
    RandomErasing, \
    ToPILImage

Finetune_weights_path = r"weights\psa_CIFAR10_e11_0.876.pth"

transform0 = transforms.Compose([
                transforms.Resize(256),
                RandomHorizontalFlip(),
                RandomCrop(224, 4, padding_mode='reflect'),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
instance_test_set = torchvision.datasets.CIFAR10(root="datasets", train=False, download=False,
                                                 transform=transform0)
test_loader = torch.utils.data.DataLoader(instance_test_set, batch_size=1, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PSAMIL(dataset="CIFAR10", backbone="resnet18", need_cl=False, pos_cls=1).to(device)
state = torch.load(Finetune_weights_path)
model.load_state_dict(state['model_state_dict'])
model.firsttime = state['firsttime']

visualize_clusters(model, test_loader, "PSMIL")