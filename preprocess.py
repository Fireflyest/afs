import pandas as pd
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
import math

import ecgreader

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler, normalize, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

# TOP_BASE_COLUMNS = ['Time from waking to first cigarette',
#     'Number of cigarettes previously smoked daily',
#     'Body mass index (BMI)']

# TOP_NMR_COLUMNS = ['Glucose', 'Linoleic Acid to Total Fatty Acids percentage',
#   'Cholesterol to Total Lipids in Medium VLDL percentage',
#   'Triglycerides to Total Lipids in Medium VLDL percentage',
#   'Cholesteryl Esters to Total Lipids in Very Small VLDL percentage',
#   'Cholesterol to Total Lipids in Very Small VLDL percentage',
#   'Cholesteryl Esters to Total Lipids in Medium VLDL percentage',
#   'Free Cholesterol to Total Lipids in Small VLDL percentage']

# 基础前
TOP_BASE_COLUMNS = ['Body mass index (BMI)', 'Number of cigarettes previously smoked daily',
    'Year of birth', 'Age started smoking in former smokers',
    'Age at recruitment', 'Duration of vigorous activity', 'f.4080.0.0',
    'Number of unsuccessful stop-smoking attempts',
    'Time spent watching television (TV)', 'Average weekly red wine intake',
    'Average weekly champagne plus white wine intake',
    'Diastolic blood pressure, automated reading',
    'Average weekly spirits intake', 'Age first had sexual intercourse',
    'Number of diet questionnaires completed', 'Duration of walks',
    'Time spent outdoors in winter', 'Time spend outdoors in summer',
    'Lifetime number of sexual partners', 'Fresh fruit intake',
    'Salad / raw vegetable intake', 'Cooked vegetable intake',
    'Cereal intake', 'Sleep duration',
    'Frequency of stair climbing in last 4 weeks', 'Usual walking pace',
    'Able to confide', 'Dried fruit intake', 'Nap during day',
    'Alcohol intake frequency.',
    'Frequency of walking for pleasure in last 4 weeks',
    'Drive faster than motorway speed limit', 'Cheese intake',
    'Duration walking for pleasure', 'Processed meat intake', 'Beef intake',
    'Plays computer games', 'Exposure to tobacco smoke outside home',
    'Duration of heavy DIY', 'Oily fish intake',
    'Weekly usage of mobile phone in last 3 months',
    'Length of mobile phone use', 'Getting up in morning',
    'Alcohol intake versus 10 years previously', 'Past tobacco smoking',
    'Poultry intake', 'Sex', 'Sleeplessness / insomnia',
    'Variation in diet', 'Snoring', 'Daytime dozing / sleeping',
    'Salt added to food', 'Exposure to tobacco smoke at home',
    'Alcohol usually taken with meals',
    'Hands-free device/speakerphone use with mobile phone in last 3 month',
    'Frequency of solarium/sunlamp use',
    'Light smokers, at least 100 smokes in lifetime',
    'Smoking/smokers in household', 'Ever stopped smoking for 6+ months',
    'Ever smoked', 'Current tobacco smoking']

# 血液前
TOP_NMR_COLUMNS = ['Glucose',
    'Cholesteryl Esters to Total Lipids in Very Small VLDL percentage',
    'Cholesterol to Total Lipids in Very Small VLDL percentage',
    'Phospholipids to Total Lipids in Large HDL percentage',
    'Free Cholesterol in IDL', 'Cholesteryl Esters in IDL',
    'Cholesterol to Total Lipids in Large HDL percentage',
    'Triglycerides to Total Lipids in Small VLDL percentage',
    'Cholesteryl Esters to Total Lipids in Medium VLDL percentage',
    'Cholesterol to Total Lipids in Medium VLDL percentage',
    'Cholesterol in IDL',
    'Phospholipids to Total Lipids in Medium VLDL percentage',
    'Triglycerides to Total Lipids in Very Large VLDL percentage',
    '3-Hydroxybutyrate',
    'Phospholipids to Total Lipids in Small VLDL percentage',
    'Free Cholesterol to Total Lipids in Medium VLDL percentage',
    'Cholesteryl Esters to Total Lipids in IDL percentage',
    'Free Cholesterol to Total Lipids in Large LDL percentage',
    'Cholesterol to Total Lipids in IDL percentage',
    'Triglycerides to Total Lipids in Medium VLDL percentage',
    'Cholesteryl Esters in Very Small VLDL',
    'Cholesteryl Esters to Total Lipids in Large HDL percentage',
    'Glycoprotein Acetyls',
    'Total Concentration of Branched-Chain Amino Acids (Leucine + Isoleucine + Valine)',
    'Cholesterol to Total Lipids in Small VLDL percentage', 'Tyrosine',
    'Cholesterol to Total Lipids in Large LDL percentage', 'Glutamine',
    'Albumin', 'Acetoacetate',
    'Cholesteryl Esters to Total Lipids in Very Large VLDL percentage',
    'Creatinine',
    'Free Cholesterol to Total Lipids in Small VLDL percentage',
    'Triglycerides to Total Lipids in IDL percentage',
    'Triglycerides to Total Lipids in Medium LDL percentage', 'Alanine',
    'Citrate', 'Cholesteryl Esters in Medium VLDL',
    'Triglycerides to Total Lipids in Small LDL percentage', 'Pyruvate',
    'Cholesteryl Esters to Total Lipids in Chylomicrons and Extremely Large VLDL percentage',
    'Polyunsaturated Fatty Acids to Total Fatty Acids percentage',
    'Phospholipids in IDL',
    'Free Cholesterol to Total Lipids in Very Large HDL percentage',
    'Cholesterol in Very Small VLDL',
    'Cholesterol to Total Lipids in Very Large VLDL percentage',
    'Cholesteryl Esters to Total Lipids in Large VLDL percentage', 'Valine',
    'Glycine',
    'Free Cholesterol to Total Lipids in Very Small VLDL percentage',
    'Cholesteryl Esters in Very Large HDL',
    'Saturated Fatty Acids to Total Fatty Acids percentage',
    'Concentration of Large HDL Particles',
    'Average Diameter for LDL Particles', 'Lactate',
    'Cholesterol in Large HDL',
    'Phospholipids to Total Lipids in Medium LDL percentage', 'Leucine',
    'Linoleic Acid to Total Fatty Acids percentage',
    'Cholesterol to Total Lipids in Medium LDL percentage',
    'Omega-3 Fatty Acids to Total Fatty Acids percentage', 'Histidine',
    'Omega-6 Fatty Acids to Total Fatty Acids percentage',
    'Cholesteryl Esters to Total Lipids in Large LDL percentage',
    'Phospholipids to Total Lipids in Very Small VLDL percentage',
    'Cholesteryl Esters in Large HDL', 'Free Cholesterol in Large HDL',
    'Phospholipids to Total Lipids in Medium HDL percentage',
    'Cholesterol to Total Lipids in Chylomicrons and Extremely Large VLDL percentage',
    'Monounsaturated Fatty Acids to Total Fatty Acids percentage',
    'Acetone',
    'Triglycerides to Total Lipids in Very Small VLDL percentage',
    'Phospholipids to Total Lipids in Chylomicrons and Extremely Large VLDL percentage',
    'Polyunsaturated Fatty Acids to Monounsaturated Fatty Acids ratio']

# 用来训练的基础特征
USE_BASE_COLUMNS = TOP_BASE_COLUMNS[:48]

# 用来训练的血液特征
USE_NMR_COLUMNS = TOP_NMR_COLUMNS[:48]

COLUMNS = len(USE_BASE_COLUMNS) + len(USE_NMR_COLUMNS)
# COLUMNS = 6 + 68 + 249


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

    # 数据合并
    return pd.merge(baseline_data, life_style_data), nmr_data, ground_true_data

# 如果某个特征的缺失比例超过阈值，就把该列特征去掉
def drop_nan_column(data : pd.DataFrame, thresh=0.2):
    # 剔除缺失大于阈值的列
    # print(data.shape)
    data = data.dropna(axis=1, thresh=data.shape[0] * (1 - thresh)) # (3200, 75) -> (3200, 49)
    # print(f'data.shape={data.shape}')
    return data

# 填充缺失值
def fill_nan_data(data : pd.DataFrame, value_types: dict, sample_posterior=True):

    # data.replace(-10.0, np.nan, inplace=True)

    # for column, value in data.items():
    #     if value_types[column] == 'Categorical single':
    #         value = value.to_frame()
    #         imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    #         data.loc[:,column] = imp.fit_transform(value)

    # https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer
    imp = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending',
        max_iter=100)
    imp.set_output(transform='pandas')
    data = imp.fit_transform(data)

    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # data = imp.fit_transform(data)

    # for column, value in data.items():
    #     cov = abs(value.var() / value.mean())
    #     print(f'coefficient_of_variation={cov}')
    #     if cov > 15:
    #         value = value.to_frame()
    #         le = LabelEncoder()
    #         data.loc[:,column] = le.fit_transform(value)

    return data


# 填充缺失值
def fill_nan_data_knn(data : pd.DataFrame):
    imp = KNNImputer(n_neighbors=10, weights="distance")
    imp.set_output(transform='pandas')
    data = imp.fit_transform(data)
    return data

# 正则化
def normalize_data(data : pd.DataFrame):
    # standard_scaler = StandardScaler()
    # data = standard_scaler.fit_transform(data)

    # https://scikit-learn.org/stable/modules/preprocessing.html#mapping-to-a-gaussian-distribution
    # quantile_transformer = QuantileTransformer(random_state=0)
    # quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
    # quantile_transformer.set_output(transform='pandas')
    # data = quantile_transformer.fit_transform(data)

    # # 把数据缩放
    min_max_scaler = MinMaxScaler()
    min_max_scaler.set_output(transform='pandas')
    data = min_max_scaler.fit_transform(data)

    # 把数据缩放
    # https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
    # max_abs_scaler = MaxAbsScaler()
    # data = max_abs_scaler.fit_transform(data)

    # 把数据缩放
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)

    # l2正则化
    # https://scikit-learn.org/stable/modules/preprocessing.html#normalization
    # data = normalize(data, norm='l2')

    # data = data.astype(np.double)
    # for column, value in data.items():
    #     value = value.to_frame()

    #     scaler = MinMaxScaler().fit(value)
    #     # scaler = QuantileTransformer(output_distribution='normal', random_state=0).fit(value)

    #     data.loc[:,column] = scaler.transform(value)

    return data

# 加载数据
def load_data(path):
    field_names, value_types = read_dictionary_data()
    base_data, nmr_data, ground_true_data = read_data(path)

    eids = ground_true_data['f.eid']
    ground_true_data.drop('f.eid', axis=1, inplace=True)

    # 基础数据
    base_data.drop('f.eid', axis=1, inplace=True)
    base_data.rename(columns=field_names, inplace=True)
    base_data = base_data[USE_BASE_COLUMNS]
    processed_base_data = pd.DataFrame(normalize_data(fill_nan_data(base_data, value_types)))
    # processed_base_data = pd.DataFrame(fill_nan_data(base_data, value_types))
    # processed_base_data = base_data

    # 血液数据
    nmr_data.drop('f.eid', axis=1, inplace=True)
    nmr_data.rename(columns=field_names, inplace=True)
    nmr_data = nmr_data[USE_NMR_COLUMNS]
    processed_nmr_data = pd.DataFrame(normalize_data(nmr_data))
    # processed_nmr_data = pd.DataFrame(fill_nan_data_knn(nmr_data))
    # processed_nmr_data = nmr_data

    # total_data_columns = len(processed_base_data.columns)
    # total_data_columns = len(processed_nmr_data.columns)
    total_data_columns = len(processed_base_data.columns) + len(processed_nmr_data.columns)
    # total_data = processed_base_data
    # total_data = processed_nmr_data
    total_data = pd.concat([processed_base_data, processed_nmr_data], axis=1)
    total_data = pd.DataFrame(fill_nan_data_knn(total_data))

    total_data = pd.concat([eids, total_data, ground_true_data], axis=1)
    # total_data = total_data.sample(frac=1)

    eids = total_data['f.eid']
    total_data.drop('f.eid', axis=1, inplace=True)

    # 临时保存
    X = total_data.iloc[:,:total_data_columns]
    X.to_csv(f'.{path}/X.csv')
    y = total_data.iloc[:,total_data_columns:]
    y.to_csv(f'.{path}/y.csv')

    # eids， 特征名称， 基础数据和血液数据， ground_true
    return eids, X.columns, X, y

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
            ax.plot(coeffs_ecg(ecg_data[head]), linewidth=0.5)
            ax.plot(ecg_data[head], linewidth=0.5)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        plt.savefig(f'.{path}/ecg_temp/ecg_{eid}_processed.png', dpi=600)
        if index == 1 : break

if __name__ == '__main__':

    convert_all_ecg('/dataset/train')

    # 以下代码是对特征的重要性进行排序
    eid, column, total_data, ground_true_data = load_data('/dataset/train')
    
    X_train, X_val, y_train, y_val = train_test_split(total_data, ground_true_data.T2D, test_size=0.2, random_state=42)

    pass

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
    forest_importances = forest_importances.sort_values(ascending=False)[0:74]
    print(forest_importances)

    fig, ax = plt.subplots()
    forest_importances.plot.barh(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('./out/feature_importances.png')

    print(forest_importances.keys())

    pass