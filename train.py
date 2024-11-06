import preprocess
import module

import time

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

def train_model(model, dataloaders, dataset_sizes, loss_fc, optimizer, num_epochs=5):
    since = time.time()
    train_loss_y = []
    train_acc_y = []
    val_loss_y = []
    val_acc_y = []

    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for (inputs, ecg_image), labels in dataloaders[phase]:
                inputs = inputs.to(device)
                ecg_image = ecg_image.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(f'labels={labels}')
                    outputs = torch.squeeze(model(inputs, ecg_image))
                    # print(f'outputs={outputs}')
                    preds = torch.where(outputs > 0.5, 1.0, 0.0)
                    # print(f'preds={preds}')
                    loss = loss_fc(outputs, labels)
                    if epoch == num_epochs - 1:
                        print('------------------------------------------')
                        print(f'labels={labels}')
                        print(f'preds ={preds}')
                        print(f'outputs={outputs}')
                        print(f'sum={torch.sum(preds == labels).item()}')
                        fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), preds.cpu().numpy())
                        roc_auc = auc(fpr, tpr)
                        print(f'roc_auc={roc_auc}')
                        pass
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print(torch.sum(preds == labels))
                # running_corrects += torch.sum(preds == labels) / 2.0
                running_corrects += torch.sum(preds == labels)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                train_loss_y.append(float(epoch_loss))
                train_acc_y.append(float(epoch_acc))
            if phase == 'val':
                val_loss_y.append(float(epoch_loss))
                val_acc_y.append(float(epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "./out/best.pt")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    torch.save(model.state_dict(), "./out/last.pt")
    return model, train_loss_y, train_acc_y, val_loss_y, val_acc_y

if __name__ == '__main__':

    device = (
        "cuda:3"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    transform = transforms.Compose([
        # transforms.CenterCrop(224), # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    path = '/dataset/train'
    eids, column, total_data, ground_true_data = preprocess.load_data(path)

    # 心电图xml文件转化为图片并返回图片的路径
    ecg_paths = { index : preprocess.convert_ecg(path, eid) for index, eid in enumerate(eids.to_list()) }

    # 划分数据集
    train_dataset = module.DiabetesDataset(transform, total_data, ground_true_data, ecg_paths)
    train_data_size = int(0.8 * len(train_dataset))
    val_data_size = len(train_dataset) - train_data_size
    train_data, val_data = random_split(train_dataset, [train_data_size, val_data_size])

    # 数据加载器
    dataloaders = { 'train' : DataLoader(train_data, batch_size=8, shuffle=True), 'val' : DataLoader(val_data, batch_size=8, shuffle=False) }
    dataset_sizes = { 'train' : train_data_size, 'val' : val_data_size }

    # 模型
    model = module.DiabetesPredictNet().to(device)
    model.apply(module.initialize_weights)

    # 损失函数
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()

    # 优化器
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    model, train_loss_y, train_acc_y, val_loss_y, val_acc_y = train_model(
        model, 
        dataloaders, 
        dataset_sizes, 
        criterion, 
        optimizer, 
        num_epochs=100
    )

    plt.plot(train_acc_y, label="train_acc")
    plt.plot(val_acc_y, label="val_acc", linestyle=':')
    plt.legend(loc='lower right')
    plt.savefig('./out/acc.png')

    plt.plot(train_loss_y, label="train_loss")
    plt.plot(val_loss_y, label="val_loss", linestyle=':')
    plt.legend(loc='lower right')
    plt.savefig('./out/loss.png')
