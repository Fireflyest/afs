import preprocess
import module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve,f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE


# 使用机器学习的方法训练和预测

# Utility function to train and evaluate a model on PIMA dataset
def train_and_evaluate_model(model,X,y, verbose=False, n_splits=10):
    acc = 0
    auc = 0
    f1 = 0
    prec = 0
    recall = 0
    
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=123)
    
    start_time = time.time()
    for train_index, test_index in sss.split(X, y):
        if X is pd.DataFrame:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc1 = accuracy_score(y_test, y_pred)
        auc1 = roc_auc_score(y_test, y_pred, average="macro")
        f11 = f1_score(y_test, y_pred, average="macro")
        prec1 = precision_score(y_test, y_pred, average="macro")
        recall1 = recall_score(y_test, y_pred, average="macro")
        if verbose:
            print ('acc', acc1)
            print('f1', f11)
            print('recall1', recall1)
            print('auc1', auc1)
        acc += acc1
        auc += auc1
        f1 += f11
        prec += prec1
        recall += recall1
        
    spent_time = time.time() - start_time
    print("Acc      F-Meas   Precis   Recall   AUC      Time")
    print("%.04f\t%.04f\t%.04f\t%.04f\t%.04f\t%0.4f" % (acc/n_splits, f1/n_splits, prec/n_splits, 
                                            recall/n_splits, auc/n_splits, spent_time))
    



if __name__ == '__main__':

    # 加载数据并预处理
    path = '/dataset/train'
    eids, column, total_data, ground_true_data = preprocess.load_data(path)

    # 此处无用，深度学习卷积用的
    transform = transforms.Compose([
        # transforms.CenterCrop(224), # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 心电图xml文件转化为图片并返回图片的路径
    ecg_paths = { index : preprocess.convert_ecg(path, eid) for index, eid in enumerate(eids.to_list()) }
    
    # 数据读取，在module文件下设置取前几个特征
    train_dataset = module.DiabetesDataset(transform, total_data, ground_true_data, ecg_paths)

    # X = pd.DataFrame(train_dataset.X.numpy())
    # y = pd.DataFrame(train_dataset.y.numpy())
    X = total_data.values
    y = ground_true_data.T2D.values
    # oversample = SMOTE(random_state=123)
    # X, y = oversample.fit_resample(X, y)   

    # 数据集内的数据取出划分
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



    # from collections import OrderedDict
    # # NOTE: Setting the `warm_start` construction parameter to `True` disables
    # # support for parallelized ensembles but is necessary for tracking the OOB
    # # error trajectory during training.
    # ensemble_clfs = [
    #     (
    #         "RandomForestClassifier, max_features='sqrt'",
    #         RandomForestClassifier(
    #             warm_start=True,
    #             oob_score=True,
    #             max_features="sqrt",
    #             random_state=42,
    #         ),
    #     ),
    #     (
    #         "RandomForestClassifier, max_features='log2'",
    #         RandomForestClassifier(
    #             warm_start=True,
    #             max_features="log2",
    #             oob_score=True,
    #             random_state=42,
    #         ),
    #     ),
    #     (
    #         "RandomForestClassifier, max_features=None",
    #         RandomForestClassifier(
    #             warm_start=True,
    #             max_features=None,
    #             oob_score=True,
    #             random_state=42,
    #         ),
    #     ),
    # ]

    # # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    # error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # # Range of `n_estimators` values to explore.
    # min_estimators = 15
    # max_estimators = 150

    # for label, clf in ensemble_clfs:
    #     for i in range(min_estimators, max_estimators + 1, 5):
    #         clf.set_params(n_estimators=i)
    #         clf.fit(X, y)

    #         # Record the OOB error for each `n_estimators=i` setting.
    #         oob_error = 1 - clf.oob_score_
    #         error_rate[label].append((i, oob_error))

    # # Generate the "OOB error rate" vs. "n_estimators" plot.
    # for label, clf_err in error_rate.items():
    #     xs, ys = zip(*clf_err)
    #     plt.plot(xs, ys, label=label)

    # plt.xlim(min_estimators, max_estimators)
    # plt.xlabel("n_estimators")
    # plt.ylabel("OOB error rate")
    # plt.legend(loc="upper right")
    # plt.savefig('./out/RandomForestClassifier.png')





    # # Create tuned model for RF
    # rf_tuned_nofs_nobl = RandomForestClassifier(max_depth = 5,  max_features = None, 
    #                         criterion = 'entropy', n_estimators = 100, random_state=123)

    # # Train and evaluate the Random Forest Model
    # train_and_evaluate_model(rf_tuned_nofs_nobl, X_train, y_train, verbose=True)

    # # Do prediction (example) with the trained model
    # y_pred = rf_tuned_nofs_nobl.predict(X_val)

    # print((y_pred + y_val))

    # # if We have the labels with can also evaluate the prediction        
    # acc1 = accuracy_score(y_val, y_pred)
    # auc1 = roc_auc_score(y_val, y_pred, average="macro")
    # f11 = f1_score(y_val, y_pred, average="macro")
    # prec1 = precision_score(y_val, y_pred, average="macro")
    # recall1 = recall_score(y_val, y_pred, average="macro")
    # print ('acc', acc1)
    # print('f1', f11)
    # print('recall1', recall1)
    # print('auc1', auc1)



    # 决策树
    clf1 = DecisionTreeClassifier(max_depth=4, min_samples_split=2,
    random_state=0, max_features='log2')
    # clf1.fit(X_train, y_train)
    train_and_evaluate_model(clf1, X_train, y_train, verbose=False)
    y_pred1 = clf1.predict(X_val)
    print(accuracy_score(y_pred1, y_val))

    # 随机森林
    clf2 = RandomForestClassifier(n_estimators=80, max_depth=None,
        min_samples_split=2, random_state=0, max_features="sqrt",)
    # clf2.fit(X_train, y_train)
    train_and_evaluate_model(clf2, X_train, y_train, verbose=False)
    y_pred2 = clf2.predict(X_val)
    print(accuracy_score(y_pred2, y_val))

    # 极端决策树
    clf3 = ExtraTreesClassifier(n_estimators=80, max_depth=None,
        min_samples_split=2, random_state=0)
    # clf3.fit(X_train, y_train)
    train_and_evaluate_model(clf3, X_train, y_train, verbose=False)
    y_pred3 = clf3.predict(X_val)
    print(accuracy_score(y_pred3, y_val))

    # 集成训练，用多个决策树来集成一个大模型，这里用100个
    clf4 = AdaBoostClassifier(n_estimators=100, algorithm="SAMME")
    # clf4.fit(X_train, y_train)
    train_and_evaluate_model(clf4, X_train, y_train, verbose=False)
    y_pred4 = clf4.predict(X_val)
    print(accuracy_score(y_pred4, y_val))

    # 软投票
    eclf = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('et', clf3), ('ab', clf4)],
                            voting='soft', weights=[1, 1, 1, 1])
    # eclf.fit(X_train, y_train)
    train_and_evaluate_model(eclf, X_train, y_train, verbose=False)
    y_pred5 = eclf.predict(X_val)
    print(accuracy_score(y_pred5, y_val))

    y_pred = y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5
    print(y_pred/6 + y_val)
    y_pred = np.where(y_pred >= 3, 1, 0)
    acc1 = accuracy_score(y_val, y_pred)
    auc1 = roc_auc_score(y_val, y_pred, average="macro")
    f11 = f1_score(y_val, y_pred, average="macro")
    prec1 = precision_score(y_val, y_pred, average="macro")
    recall1 = recall_score(y_val, y_pred, average="macro")
    print('auc1', auc1)
    print ('acc', acc1)
    print('f1', f11)
    print('prec1', prec1)
    print('recall1', recall1)







    # # 集成训练，用多个决策树来集成一个大模型，这里用100个
    # clf1 = AdaBoostClassifier(n_estimators=100, algorithm="SAMME")
    # clf1.fit(X_train, y_train)
    # y_pred = clf1.predict(X_val)
    # fpr, tpr, thresholds = roc_curve(y_pred, y_val)
    # roc_auc = auc(fpr, tpr)
    # print(f'roc_auc={roc_auc}')
    # print(f'acc={accuracy_score(y_pred, y_val)}')





    # # 尝试更多机器学习的方法
    # clf1 = DecisionTreeClassifier(max_depth=4)
    # clf2 = KNeighborsClassifier(n_neighbors=7)
    # clf3 = SVC(kernel='rbf', probability=True)
    # clf4 = HistGradientBoostingClassifier()
    # clf5 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    #     random_state=0)
    # eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
    #                         voting='soft', weights=[2, 1, 2])


    # for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['DecisionTreeClassifier', 'KNeighborsClassifier', 'SVC', 'VotingClassifier', 'HistGradientBoostingClassifier', 'DecisionTreeClassifier']):
    #     scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
    #     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


    # # clf1 = clf1.fit(X_train, y_train)
    # # clf2 = clf2.fit(X_train, y_train)
    # # clf3 = clf3.fit(X_train, y_train)
    # # eclf = eclf.fit(X_train, y_train)

    # print(accuracy_score(clf1.predict(X_val), y_val))
    # print(accuracy_score(clf2.predict(X_val), y_val))
    # print(accuracy_score(clf3.predict(X_val), y_val))
    # print(accuracy_score(eclf.predict(X_val), y_val))







    # # 决策树分类器
    # # max_depth决策树最大深度
    # # criterion = gini/entropy 可以用来选择用基尼指数或者熵来做损失函数。
    # base_model = DecisionTreeClassifier(max_depth=1, criterion='gini',random_state=1).fit(X_train, y_train)
    # y_pred = base_model.predict(X_val)  # 预测模型结果
    # print(f"决策树的准确率：{accuracy_score(y_val, y_pred):.3f}")

    # model = AdaBoostClassifier(
    #                             n_estimators=50,
    #                             learning_rate=0.4,
    #                             algorithm='SAMME',
    #                             random_state=1)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_val)
    # print(f"AdaBoost的准确率：{accuracy_score(y_val,y_pred):.3f}")

    # # 测试估计器个数的影响
    # x = list(range(2, 102, 2))
    # y = []

    # for i in x:
    #     model = AdaBoostClassifier(
    #                             n_estimators=i,
    #                             learning_rate=0.4,
    #                             algorithm='SAMME',
    #                             random_state=1)
    #     model.fit(X_train, y_train)
    #     model_test_sc = accuracy_score(y_val, model.predict(X_val))
    #     y.append(model_test_sc)

    # plt.style.use('ggplot')
    # plt.title("Effect of n_estimators", pad=20)
    # plt.xlabel("Number of base estimators")
    # plt.ylabel("Test accuracy of AdaBoost")
    # plt.plot(x, y)
    # plt.savefig('./out/ada.png')

    # 使用GridSearchCV自动调参
    # GridSearch和CV，即网格搜索和交叉验证。
    # 网格搜索，搜索的是参数，即在指定的参数范围内，按步长依次调整参数，利用调整的参数训练学习器，从所有的参数中找到在验证集上精度最高的参数，这其实是一个训练和比较的过程。
    # hyperparameter_space = {'n_estimators':list(range(2, 102, 2)),
    #                         'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

    # gs = GridSearchCV(AdaBoostClassifier(
    #                                     algorithm='SAMME',
    #                                     random_state=1),
    #                 param_grid=hyperparameter_space,
    #                 scoring="accuracy", n_jobs=-1, cv=5)

    # gs.fit(X_train, y_train)
    # print("最优超参数:", gs.best_params_)
