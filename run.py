import preprocess
import module
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import time

def test_model(model, dataloader, batch_size):
    model.eval()
    with torch.no_grad():
        for (inputs, ecg_image), labels in dataloader:
            inputs = inputs.to(device)
            ecg_image = ecg_image.to(device)
            labels = labels.to(device)
            outputs = torch.squeeze(model(inputs, ecg_image))
            preds = torch.where(outputs > 0.5, 1.0, 0.0)
            fpr, tpr, thresholds = roc_curve(labels.cpu().clone().detach().numpy(), outputs.cpu().clone().detach().numpy())
            roc_auc = auc(fpr, tpr)
            print(f'labels={labels}')
            print(f'preds ={preds}')
            print(f'roc_auc={roc_auc}')
            print(f'sum={torch.sum(preds == labels).item()}')
            print(f'acc={torch.sum(preds == labels).item() / batch_size}')

            plt.cla() 
            plt.plot(fpr, tpr, label='AUC')
            plt.xlabel('False Positive Rate')  # x坐标轴标题
            plt.ylabel('True Positive Rate')  # y坐标轴标题
            plt.xlim([0.0, 1.0])  # 限定x轴的范围
            plt.ylim([0.0, 1.0])  # 限定y轴的范围
            plt.legend(loc='lower right')
            plt.grid()
            plt.savefig('./out/auc.png')


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

    path = '/dataset/test'
    eids, column, total_data, ground_true_data = preprocess.load_data(path)

    # 心电图xml文件转化为图片并返回图片的路径
    ecg_paths = { index : preprocess.convert_ecg(path, eid) for index, eid in enumerate(eids.to_list()) }

    # 划分数据集
    dataset = module.DiabetesDataset(transform, total_data, ground_true_data, ecg_paths)

    model = module.DiabetesPredictNet()
    model.load_state_dict(torch.load('./best.pt', weights_only=True))
    model.to(device)

    batch_size = int(len(dataset) / 2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_model(model, dataloader, batch_size)
