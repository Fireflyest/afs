import preprocess
import module
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

if __name__ == '__main__':

    path = '/dataset/train'
    eids, column, total_data, ground_true_data = preprocess.load_data(path)

    transform = transforms.Compose([
        # transforms.CenterCrop(224), # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 心电图xml文件转化为图片并返回图片的路径
    ecg_paths = { index : preprocess.convert_ecg(path, eid) for index, eid in enumerate(eids.to_list()) }
    # 划分数据集
    train_dataset = module.DiabetesDataset(transform, total_data, ground_true_data, ecg_paths)

    X_train, X_val, y_train, y_val = train_test_split(train_dataset.X, ground_true_data.Complication, test_size=0.2, random_state=42)

    print(X_train.shape)

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    # Plot the results
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()