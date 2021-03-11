import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
from typing import Tuple, Sequence, Callable
import albumentations as A
import os
import cv2
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.models import resnet101
from collections import OrderedDict
import random
import csv
from albumentations.core.transforms_interface import ImageOnlyTransform
from efficientnet_pytorch import EfficientNet
from torch_poly_lr_decay import PolynomialLRDecay
import torchvision.models as models

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


class config:
    seed = 42
    device = "cuda:0"
    lr = 1e-4
    epochs = 25
    batch_size = 1
    train_5_folds = True


seed_everything(config.seed)



class MnistDataset(Dataset):
    def __init__(
            self,
            dir: os.PathLike,
            image_ids: os.PathLike,
            transforms: Sequence[Callable],
            albumentations: Sequence[Callable],
    ) -> None:
        self.dir = dir
        self.transforms = transforms
        self.albumentations = albumentations

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        img = cv2.imread(
            os.path.join(self.dir, f'{str(image_id).zfill(5)}.png'))
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.albumentations is not None:
            transformed = self.albumentations(image=img)
            transformed_image = transformed["image"]
            if self.transforms is not None:
                image = self.transforms(transformed_image)
        else:
            if self.transforms is not None:
                image = self.transforms(img)

        return image, target

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class Resnet101_32x8d(nn.Module):
    def __init__(self, num_classes: int = 26) -> None:
        super().__init__()
        self.model_ft = models.resnext101_32x8d(pretrained=True)
        self.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1000, 26)),
                                                      ('output', nn.Sigmoid())
                                                    ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.model_ft(x)
        x = self.classifier(x)
        return x

class EfficientNetb3(nn.Module):
    def __init__(self, num_classes: int = 26) -> None:
        super().__init__()
        self.model_ft = EfficientNet.from_pretrained('efficientnet-b3')
        self.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1000, 26)),
                                                      ('output', nn.Sigmoid())
                                                    ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.model_ft(x)
        x = self.classifier(x)
        return x


def load_model(model_name, path, device: torch.device = 'cuda') -> nn.Module:
    if model_name == 'Resnet101_32x8d':
        model = Resnet101_32x8d().to(device)
    if model_name == 'EfficientNetb3':
        model = EfficientNetb3().to(device)
    model.load_state_dict(torch.load(path))
    return model

testset = MnistDataset('./data/test', f'./data/sample_submission.csv', transforms_test, None)
test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)

def test(device: torch.device = 'cuda'):
    submit = pd.read_csv('./data/sample_submission.csv')


    model1 = load_model('Resnet101_32x8d', './model/resnet101_32x8d-f0-15.pth')
    model2 = load_model('Resnet101_32x8d', './model/resnet101_32x8d-f1-18.pth')
    model3 = load_model('Resnet101_32x8d', './model/resnet101_32x8d-f2-18.pth')
    model4 = load_model('Resnet101_32x8d', './model/resnet101_32x8d-f3-18.pth')
    model5 = load_model('Resnet101_32x8d', './model/resnet101_32x8d-f4-18.pth')

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    outputs = 0
    batch_size = test_loader.batch_size
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)

        outputs1 = model1(images)
        outputs2 = model2(images)
        outputs3 = model3(images)
        outputs4 = model4(images)
        outputs5 = model5(images)

        outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5)/5
        batch_index = i * batch_size
        outputs = outputs >= 0.5
        predict = outputs.long().squeeze(0).detach().cpu().numpy()
        submit.iloc[batch_index:batch_index+batch_size, 1:] = predict
        # submit.iloc[:, 1:] = np.where((submit.values[:,1:]/3)>=0.5, 1,0)
    submit.to_csv('submit.csv', index=False)


if __name__ == '__main__':
    test()