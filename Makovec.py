import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

TRAIN_PATH = 'archive/train/'
TEST_PATH = 'archive/test/'
CAT_PATH = 'cats/'
DOG_PATH = 'dogs/'
PANDA_PATH = 'panda/'
CLASS_NAMES = ['Cat', 'Dog', 'Panda']
RESIZE_W = 128
RESIZE_H = 128
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
NEPOCH = 1
LOAD_PRETRAINED = False
USE_CUDA = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() and USE_CUDA else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
                # Сверточные слои
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=0)
        
        # MaxPooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        
        # Вычисляем размерность выхода после сверток и пула
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, RESIZE_W, RESIZE_H)
            out = self.conv1(dummy_input)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.pool(out)
            self.flatten_size = out.numel()
        
        # Полносвязный слой
        self.fc = nn.Linear(self.flatten_size, 3)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Разворачиваем тензор
        x = self.fc(x)
        return x

def readImage(filepath : str):
    if os.path.isfile(filepath) and filepath.endswith('.jpg'):
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] < RESIZE_H or img.shape[1] < RESIZE_W:
                img = cv2.resize(img, (RESIZE_H, RESIZE_W), interpolation=cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (RESIZE_H, RESIZE_W), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img
    return None

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path : str):
        super().__init__()
        self.data_list = []
        self.readImages(os.path.join(path, CAT_PATH), 0)
        self.readImages(os.path.join(path, DOG_PATH), 1)
        self.readImages(os.path.join(path, PANDA_PATH), 2)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
    
    def readImages(self, path : str, class_id : int):
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            img = readImage(filepath)
            if img is not None:
                self.data_list.append((img, class_id,))
        

def plotTensor(img):
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()

def eval_metrics(model : Net, dataset : CustomDataset):
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        model.eval()
        loss_val, acc_val = 0., 0.
        for (imgs, labels,) in tqdm(dataloader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            answer = model(imgs)
            loss = F.cross_entropy(answer, labels, reduction='sum')
            loss_val += loss.detach().cpu().item()
            acc_val += (answer.detach().cpu().argmax(1) == labels.detach().cpu()).sum().item()
        loss_val = loss_val / len(dataset)
        acc_val = acc_val / len(dataset)
        return (loss_val, acc_val,)

def train_network(model : Net, train_dataset : CustomDataset, test_dataset : CustomDataset):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss, best_acc = eval_metrics(model, test_dataset)
    print('  TEST(loss: {}, accuracy: {})'.format(best_loss, best_acc))
    '''TRANING'''
    print('Start training...')
    for epoch in range(0, NEPOCH):
        print('Epoch ({}/{}):'.format(epoch + 1, NEPOCH))
        model.train()
        train_loss, train_acc = 0., 0.
        for (imgs, labels,) in tqdm(train_dataloader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            answer = model(imgs)
            loss = F.cross_entropy(answer, labels, reduction='sum')
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()
            train_acc += (answer.detach().cpu().argmax(1) == labels.detach().cpu()).sum().item()
        train_loss = train_loss / len(train_dataset)
        train_acc = train_acc / len(train_dataset)
        print('  TRAIN(loss: {}, accuracy: {})'.format(train_loss, train_acc))
        test_loss, test_acc = eval_metrics(model, test_dataset)
        print('  TEST(loss: {}, accuracy: {})'.format(test_loss, test_acc))
        if best_acc < test_acc:
            best_loss, best_acc = test_loss, test_acc
            torch.save(model, 'best.pth')
    print('  BEST(loss: {}, accuracy: {})'.format(best_loss, best_acc))
    
if __name__ == '__main__':
    if LOAD_PRETRAINED and os.path.exists('best.pth'):
        model = torch.load('best.pth')
        print('best.pth loaded')
    else:
        model = Net()
    model = model.to(DEVICE)
    train_network(model, CustomDataset(TRAIN_PATH), CustomDataset(TEST_PATH))