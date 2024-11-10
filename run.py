import preprocess
import module
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve,f1_score, precision_score, recall_score

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import time
import numpy as np





# 用测试集数据来得出结果
def test_model(model, dataloader, batch_size):
    model.eval()
    with torch.no_grad():
        for (inputs, ecg_image), labels in dataloader:
            inputs = inputs.to(device)
            ecg_image = ecg_image.to(device)
            labels = labels.to(device)
            outputs = torch.squeeze(model(inputs, ecg_image))
            preds = torch.where(outputs > 0.5, 1.0, 0.0)
            print(f'result={labels + preds/2}')

            y_val = labels.cpu().clone().detach().numpy()
            y_pred = outputs.cpu().clone().detach().numpy()
            fpr, tpr, thresholds = roc_curve(y_val, y_pred)
            roc_auc = auc(fpr, tpr)
            y_pred = np.where(y_pred > 0.5, 1.0, 0.0)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")
            prec = precision_score(y_val, y_pred, average="macro")
            recall = recall_score(y_val, y_pred, average="macro")
            print(f'roc_auc={roc_auc}')
            print ('acc', acc)
            print('f1', f1)
            print('prec', prec)
            print('recall', recall)

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
        "cuda:1"
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

    # 加载模型
    model = module.DiabetesPredictNet()
    model.load_state_dict(torch.load('./out/best.pt', weights_only=True))
    # model.load_state_dict(torch.load('./out/last.pt', weights_only=True))
    model.to(device)

    # 取一半因为内存不够
    batch_size = int(len(dataset) / 2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_model(model, dataloader, batch_size)
