import pandas as pd
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt

import ecgreader

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 心电数据用不同颜色画，部位接近的颜色比较接近
HEADS = {
    'I' : 'red', 
    'II' : 'crimson', 
    'III' : 'brown', 
    'aVR' : 'forestgreen', 
    'aVL' : 'limegreen', 
    'aVF' : 'seagreen', 
    'V1' : 'dodgerblue', 
    'V2' : 'cornflowerblue', 
    'V3' : 'royalblue', 
    'V4' : 'mediumblue', 
    'V5' : 'blue', 
    'V6' : 'darkblue'
}

# 读取特征对应的名称及其类型
def read_dictionary_data():
    dictionary_io = r'./dataset/Data_Dictionary.xlsx'
    # 数据字典
    dictionary_data: pd.DataFrame = pd.read_excel(io=dictionary_io) # 322类
    # print(dictionary_data.shape)
    field_names = dictionary_data[['FieldID', 'Field']].set_index('FieldID').to_dict(orient='dict')['Field']
    value_types = dictionary_data[['Field', 'ValueType']].set_index('Field').to_dict(orient='dict')['ValueType']
    # 缺失
    value_types['f.4080.0.0'] = 'Integer'
    return field_names, value_types

# 读取所有数据
def read_data(path):
    baseline_io = f'.{path}/Baseline_characteristics.xlsx'
    life_style_io = f'.{path}/life_style.xlsx'
    nmr_io = f'.{path}/NMR.xlsx'
    ground_true_io = f".{path}/ground_true.xlsx"

    baseline_data = pd.read_excel(io=baseline_io) # (3200, 7)
    # print(baseline_data.shape)
    life_style_data = pd.read_excel(io=life_style_io) # (3200, 69)
    # print(life_style_data.shape)
    nmr_data = pd.read_excel(io=nmr_io) # (3200, 250) 
    # print(nmr_data.shape)
    ground_true_data = pd.read_excel(io=ground_true_io) # (3200, 4) 
    # print(ground_true_data.shape)

    # TODO 血液数据的缺失比较特殊，一般是某个人的一整行数据全空
    nmr_data.fillna(nmr_data.mean(), inplace=True)
    # nmr_data.fillna(0, inplace=True)

    # 数据合并
    return pd.merge(baseline_data, life_style_data), nmr_data, ground_true_data

# 如果某个特征的缺失比例超过阈值，就把该列特征去掉
def drop_nan_column(data : pd.DataFrame, thresh=0.0):
    # 剔除缺失大于阈值的列
    # print(data.shape)
    data = data.dropna(axis=1, thresh=data.shape[0] * (1 - thresh)) # (3200, 75) -> (3200, 49)
    # print(f'data.shape={data.shape}')
    return data

# 填充缺失值
def fill_nan_data(data : pd.DataFrame):
    # https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer
    imp = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')
    data = imp.fit_transform(data)

    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # data = imp.fit_transform(data)
    return data

# 正则化
def normalize_data(data):
    # standard_scaler = StandardScaler()
    # data = standard_scaler.fit_transform(data)

    # https://scikit-learn.org/stable/modules/preprocessing.html#mapping-to-a-gaussian-distribution
    # quantile_transformer = QuantileTransformer(random_state=0)
    # quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
    # data = quantile_transformer.fit_transform(data)

    # 把数据缩放在0~1
    # min_max_scaler = MinMaxScaler()
    # data = min_max_scaler.fit_transform(data)

    # 数据正态分布在y轴附近
    # https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
    max_abs_scaler = MaxAbsScaler()
    data = max_abs_scaler.fit_transform(data)

    # l2正则化
    # https://scikit-learn.org/stable/modules/preprocessing.html#normalization
    # data = normalize(data, norm='l2')
    return data

# 加载数据
def load_data(path):
    field_names, value_types = read_dictionary_data()
    base_data, nmr_data, ground_true_data = read_data(path)

    eid = base_data['f.eid']

    ground_true_data.drop('f.eid', axis=1, inplace=True)

    # 基础数据
    base_data.rename(columns=field_names, inplace=True)
    base_data.drop('f.eid', axis=1, inplace=True)
    processed_base_data = pd.DataFrame(normalize_data(fill_nan_data(drop_nan_column(base_data, thresh=0.2))))
    # processed_base_data = pd.DataFrame(fill_nan_data(drop_nan_column(base_data, thresh=0.2)))
    processed_base_data.rename(columns={ i : base_data.columns.to_list()[i] for i in range(0, len(base_data.columns)) }, inplace=True)
    # print(processed_base_data)

    # 血液数据
    nmr_data.rename(columns=field_names, inplace=True)
    nmr_data.drop('f.eid', axis=1, inplace=True)
    processed_nmr_data = pd.DataFrame(normalize_data(nmr_data))
    # processed_nmr_data = pd.DataFrame(nmr_data)
    processed_nmr_data.rename(columns={ i : nmr_data.columns.to_list()[i] for i in range(0, len(nmr_data.columns)) }, inplace=True)
    # print(processed_nmr_data)

    # total_data = processed_base_data
    # total_data = nmr_data
    total_data = pd.concat([processed_base_data, processed_nmr_data], axis=1)

    # 临时保存
    temp_io = f'.{path}/Temp.csv'
    total_data.to_csv(temp_io)

    # eid， 特征名称， 基础数据和血液数据， ground_true
    return eid, total_data.columns, total_data, ground_true_data

# 从xml文件中读取心电数据
def read_ecg(path):
    reader = ecgreader.CardioSoftECGXMLReader(path) #Create a CardioSoftECGXMLreader class
    return reader.LeadVoltages # Extract voltages as an array of size (samplesize, 12)

# 小波分析，把曲线变平滑
def coeffs_ecg(data):
    coeffs = pywt.wavedec(data, 'db8')
    coeffs_thresh = map(lambda x: pywt.threshold(x, 50, mode='hard'), coeffs)
    return pywt.waverec(list(coeffs_thresh), 'db8')

# 把心电数据转化为图片
def convert_ecg(path, eid):
    os.makedirs(f'.{path}/ecg_temp', exist_ok=True)
    save_path = f'.{path}/ecg_temp/{eid}.png'

    plt.cla()
    plt.close()

    ecg_xml_path = f'.{path}/ecg/{eid}_20205_2_0.xml'
    if not os.path.exists(save_path):
        if os.path.exists(ecg_xml_path):
            ecg_data = read_ecg(ecg_xml_path)
            fig = plt.figure()
            # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
            for index, (head, color) in enumerate(HEADS.items()):
                ax = fig.add_subplot(12, 1, index+1)
                ax.plot(coeffs_ecg(ecg_data[head]), linewidth=0.3, color=color)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        print(f'save ecg image to {save_path}')
        plt.savefig(save_path)
    return save_path

# 转化所有心电数据，测试用的方法
def convert_all_ecg(path):
    ecg_dir = f'.{path}/ecg'
    os.makedirs(f'.{path}/ecg_temp', exist_ok=True)
    # 获取当前目录下的所有文件
    files = [os.path.join(ecg_dir, file) for file in os.listdir(ecg_dir)]
    # 遍历文件列表，输出文件名
    for index, file in enumerate(files):
        eid = os.path.splitext(os.path.basename(file))[0][:7]
        ecg_data = read_ecg(file)
        plt.cla()
        plt.close()
        fig = plt.figure()
        # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        for index, head in enumerate(HEADS):
            ax = fig.add_subplot(12, 1, index+1)
            ax.plot(coeffs_ecg(ecg_data[head]), linewidth=0.3)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        plt.savefig(f'.{path}/ecg_temp/ecg_{eid}_processed.png', dpi=600)
        # if index == 10 : break

if __name__ == '__main__':


    # convert_all_ecg('/dataset/train')




    # 以下代码是对特征的重要性进行排序
    eid, column, total_data, ground_true_data = load_data('/dataset/train')
    
    X_train, X_val, y_train, y_val = train_test_split(total_data, ground_true_data.T2D, test_size=0.2, random_state=42)

    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#feature-importance-based-on-feature-permutation
    feature_names = column.to_list()
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances = forest_importances.sort_values(ascending=False)[0:64]
    print(forest_importances)

    fig, ax = plt.subplots()
    forest_importances.plot.barh(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('./out/feature_importances.png')

    print(forest_importances.keys())

    pass