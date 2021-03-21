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


def diagonal_reverse(img):
    transformed_img = img.copy()
    center = img.shape[0] // 2
    transformed_img[0:center, 0:center] = img[center:center + center, center:center + center]
    transformed_img[0:center, center:center + center] = img[center:center * 2, 0:center]
    transformed_img[center:center + center, 0:center] = img[0:center, center:center * 2]
    transformed_img[center:center + center, center:center + center] = img[0:center, 0:center]

    return transformed_img


dirty_mnist_answer = pd.read_csv("./data/dirty_mnist_2nd_answer.csv")
# %%

dirty_mnist_answer.head()
# %%

dirty_mnist_answer[list(dirty_mnist_answer.columns.values)].describe()


# %%
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
    epochs = 80
    batch_size = 1
    train_5_folds = True


seed_everything(config.seed)


# %%

def split_dataset(path: os.PathLike) -> None:
    df = pd.read_csv(path)
    kfold = KFold(n_splits=5)
    print(kfold)
    for fold, (train, valid) in enumerate(kfold.split(df, df.index)):
        df.loc[valid, 'kfold'] = int(fold)

    df.to_csv('./data/split_kfold.csv', index=False)


class DiagonalReverse(ImageOnlyTransform):

    def __init__(
            self,
            always_apply=False,
            p=1
    ):
        super(DiagonalReverse, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return diagonal_reverse(img)


# %%

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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


# %%

transforms_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

a_train = A.Compose([
    A.HorizontalFlip(0.5),
    A.RandomRotate90(0.5),
    A.VerticalFlip(0.5),
    A.OneOf([
       A.GaussNoise(var_limit=[10, 50]),
       A.MotionBlur(),
       A.MedianBlur(),
   ], p=0.2),
   A.OneOf([
       A.OpticalDistortion(distort_limit=1.0),
       A.GridDistortion(num_steps=5, distort_limit=1.),
       A.ElasticTransform(alpha=3),
   ], p=0.2),
    A.OneOf([
      A.JpegCompression(),
  ], p=0.2),
    A.Cutout(num_holes=10, max_h_size=5, max_w_size=5, always_apply=False, p=0.5, ),
    # DiagonalReverse(),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# %%


class MnistModel(nn.Module):
    def __init__(self, num_classes: int = 26) -> None:
        super().__init__()
        self.model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        # set_parameter_requires_grad(self.model_ft, True)
        self.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1000, 26)),
                                                      # ('relu', nn.ReLU()),
                                                      # ('dropout',nn.Dropout(0.3)),
                                                      # ('fc2', nn.Linear(5000,2000)),
                                                      # ('relu', nn.ReLU()),
                                                      # ('dropout',nn.Dropout(0.3)),
                                                      # ('fc3', nn.Linear(2000, 26)),
                                                      ('output', nn.Sigmoid())
                                                    ]))

        # self._initialize_weights()


    def forward(self, x: Tensor) -> Tensor:
        x = self.model_ft(x)
        x = self.classifier(x)

        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
         for param in model.parameters():
            param.requires_grad = False
# %%

def train(fold: int, verbose: int = 100) -> None:
    split_dataset('./data/dirty_mnist_2nd_answer.csv')
    df = pd.read_csv('./data/split_kfold.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    df_train.drop(['kfold'], axis=1).to_csv(f'./data/train-kfold-{fold}.csv', index=False)
    df_valid.drop(['kfold'], axis=1).to_csv(f'./data/valid-kfold-{fold}.csv', index=False)

    trainset = MnistDataset('./data/train', f'./data/train-kfold-{fold}.csv', transforms_train, a_train)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

    validset = MnistDataset('./data/train', f'./data/valid-kfold-{fold}.csv', transforms_test, None)
    valid_loader = DataLoader(validset, batch_size=8, shuffle=False)

    num_epochs = config.epochs
    device = 'cuda'

    model = MnistModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    decay_steps = (len(trainset) // config.batch_size) * config.epochs
    scheduler = PolynomialLRDecay(optimizer, max_decay_steps=decay_steps,
                                                           end_learning_rate=1e-6, power=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.9, momentum=0.9)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)


            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % verbose == 0:
                outputs = outputs > 0.5
                acc = (outputs == targets).float().mean()
                print(f'Fold {fold} | Epoch {epoch} | Train_L: {loss.item():.7f} | Train_A: {acc.item():.7f}')

        model.eval()
        valid_acc = 0.0
        valid_loss = 0.0
        with torch.no_grad():
            for i, (images, targets) in enumerate(valid_loader):
                images = images.to(device)
                targets = targets.to(device)


                outputs = model(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                outputs = outputs > 0.5
                valid_acc += (outputs == targets).float().mean()
            print(f'Fold {fold} | Epoch {epoch} | valid_L: {valid_loss / (i + 1):.7f} | valid_A: {valid_acc / (i + 1):.7f}\n')

        if epoch > num_epochs - 10 and epoch < num_epochs - 1:
            torch.save(model.state_dict(), f'./data/efficientnet7-f{fold}-{epoch}.pth')


# %%

if __name__ == '__main__':
    train(0)
    train(1)
    train(2)
    train(3)
    train(4)