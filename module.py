import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import pandas as pd 

from PIL import Image

from torch.utils.data import Dataset


# 基础前48
TOP_BASE_COLUMNS = ['Body mass index (BMI)',
    'Frequency of walking for pleasure in last 4 weeks', 'Cheese intake',
    'Year of birth', 'Age at recruitment', 'f.4080.0.0',
    'Current tobacco smoking', 'Fresh fruit intake', 'Poultry intake',
    'Getting up in morning', 'Duration of vigorous activity',
    'Diastolic blood pressure, automated reading', 'Sleep duration',
    'Number of unsuccessful stop-smoking attempts',
    'Average weekly champagne plus white wine intake',
    'Time spent outdoors in winter', 'Ever stopped smoking for 6+ months',
    'Age started smoking in former smokers',
    'Time spent watching television (TV)', 'Cereal intake',
    'Past tobacco smoking', 'Age first had sexual intercourse', 'Calcium',
    'Light smokers, at least 100 smokes in lifetime', 'Beef intake',
    'Alcohol intake versus 10 years previously', 'Duration of walks',
    'Usual walking pace', 'Able to confide',
    'Time from waking to first cigarette', 'Former alcohol drinker',
    'Hands-free device/speakerphone use with mobile phone in last 3 month',
    'Vitamin E', 'Magnesium', 'Duration walking for pleasure',
    'Frequency of stair climbing in last 4 weeks',
    'Drive faster than motorway speed limit', 'Alcohol intake frequency.',
    'Average weekly spirits intake', 'Plays computer games', 'Ever smoked',
    'Processed meat intake', 'Salad / raw vegetable intake', 'Sex',
    'Oily fish intake', 'Number of cigarettes previously smoked daily',
    'Weekly usage of mobile phone in last 3 months',
    'Sleeplessness / insomnia', 'Englyst dietary fibre',
    'Exposure to tobacco smoke outside home',
    'Number of diet questionnaires completed', 'Variation in diet',
    'Time spend outdoors in summer', 'Salt added to food', 'Vitamin C',
    'Snoring', 'Exposure to tobacco smoke at home', 'Iron',
    'Frequency of solarium/sunlamp use', 'Length of mobile phone use',
    'Nap during day']

# 血液前48
TOP_NMR_COLUMNS = ['Glucose',
    'Cholesteryl Esters to Total Lipids in Very Small VLDL percentage',
    'Cholesterol to Total Lipids in Very Small VLDL percentage',
    'Phospholipids to Total Lipids in Large HDL percentage',
    'Free Cholesterol in IDL', 'Cholesteryl Esters in IDL',
    'Cholesterol to Total Lipids in Medium VLDL percentage',
    'Cholesterol to Total Lipids in Large HDL percentage',
    'Triglycerides to Total Lipids in Small VLDL percentage',
    'Cholesteryl Esters to Total Lipids in Medium VLDL percentage',
    'Phospholipids to Total Lipids in Medium VLDL percentage',
    'Cholesterol in IDL',
    'Triglycerides to Total Lipids in Very Large VLDL percentage',
    'Cholesteryl Esters to Total Lipids in Large HDL percentage',
    'Phospholipids to Total Lipids in Small VLDL percentage',
    'Cholesteryl Esters to Total Lipids in IDL percentage',
    '3-Hydroxybutyrate',
    'Free Cholesterol to Total Lipids in Medium VLDL percentage',
    'Free Cholesterol to Total Lipids in Small VLDL percentage',
    'Triglycerides to Total Lipids in Medium VLDL percentage',
    'Cholesterol to Total Lipids in IDL percentage',
    'Cholesterol to Total Lipids in Small VLDL percentage',
    'Cholesteryl Esters in Very Small VLDL',
    'Free Cholesterol to Total Lipids in Large LDL percentage',
    'Glycoprotein Acetyls',
    'Cholesterol to Total Lipids in Large LDL percentage', 'Tyrosine',
    'Glutamine', 'Creatinine', 'Acetoacetate',
    'Cholesteryl Esters to Total Lipids in Very Large VLDL percentage',
    'Total Concentration of Branched-Chain Amino Acids (Leucine + Isoleucine + Valine)',
    'Albumin', 'Alanine',
    'Triglycerides to Total Lipids in Medium LDL percentage',
    'Cholesteryl Esters in Medium VLDL', 'Citrate',
    'Cholesteryl Esters to Total Lipids in Chylomicrons and Extremely Large VLDL percentage',
    'Cholesteryl Esters to Total Lipids in Large VLDL percentage',
    'Triglycerides to Total Lipids in IDL percentage',
    'Cholesterol in Very Small VLDL',
    'Triglycerides to Total Lipids in Small LDL percentage', 'Valine',
    'Phospholipids in IDL', 'Pyruvate',
    'Free Cholesterol to Total Lipids in Very Small VLDL percentage',
    'Cholesteryl Esters in Very Large HDL',
    'Free Cholesterol to Total Lipids in Very Large HDL percentage',
    'Triglycerides to Total Lipids in Very Small VLDL percentage',
    'Phospholipids to Total Lipids in Medium LDL percentage',
    'Monounsaturated Fatty Acids to Total Fatty Acids percentage',
    'Lactate', 'Acetone', 'Leucine', 'Concentration of Large HDL Particles',
    'Phosphatidylcholines',
    'Cholesteryl Esters to Total Lipids in Large LDL percentage',
    'Triglycerides in Large HDL', 'Glycine', 'Total Lipids in IDL',
    'Cholesterol to Total Lipids in Very Large VLDL percentage',
    'Isoleucine', 'Omega-3 Fatty Acids to Total Fatty Acids percentage',
    'Cholesterol in Large HDL']

# 用来训练的基础特征
USE_BASE_COLUMNS = TOP_BASE_COLUMNS[:5]

# 用来训练的血液特征
USE_NMR_COLUMNS = TOP_NMR_COLUMNS[:5]

# 取多少个特征
COLUMNS = len(USE_BASE_COLUMNS) + len(USE_NMR_COLUMNS)

class DiabetesDataset(Dataset):


    def __init__(self, transform, data: pd.DataFrame, ground_true: pd.DataFrame, ecg_paths):
        super().__init__()
        self.transform = transform
        
        # 这里拼接是为了方便删除某些行
        self.data = pd.concat([data, ground_true], axis=1)
        # self.data.dropna(how='any', inplace=True, axis=0)

        # 基础数据和血液数据
        self.ecgs = ecg_paths
        # self.X = torch.tensor(self.data[TOP_COLUMNS[:COLUMNS]].values, dtype=torch.float)
        self.X = torch.tensor(self.data[USE_BASE_COLUMNS + USE_NMR_COLUMNS].values, dtype=torch.float)
        print(f'self.X={self.X}')


        # ground_true数据
        self.y = torch.tensor(self.data.T2D.values, dtype=torch.float) # 不要第一列的确诊时间
        # self.y = torch.tensor(self.data.Complication.values, dtype=torch.float) # 不要第一列的确诊时间
        # self.y = torch.tensor(self.data[['T2D','Complication']].values, dtype=torch.float) # 不要第一列的确诊时间
        print(f'self.y={self.y}')
    
    def __getitem__(self, index):
        ecg = self.transform(Image.open(self.ecgs[index]).convert('RGB'))
        return (self.X[index], ecg), self.y[index]
    
    def __len__(self):
        return len(self.X)




# 残差，自注意力，全连接
class ResidualFcBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()            
        # self.fc1 = nn.Linear(dim, dim)
        self.fc1 = SelfAttention(dim, dim, dim)
        self.dp = nn.Dropout(0.3)
        self.ac = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.dp(out)
        out = self.ac(out)
        out = self.fc2(out)
        out += identity
        out = self.ac(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # q*k的转置并除以开根号后的dim_k
        dist = torch.mm(q, k.T) * self._norm_fact
        # 归一化获得attention的相关系数
        dist = F.softmax(dist, dim=-1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        return torch.mm(dist, v)

# 总体网络
class DiabetesPredictNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basefc = ResidualFcBlock(COLUMNS)
        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], include_top=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
        self.fc = nn.Linear(512 * BasicBlock.expansion, COLUMNS)
        self.linear_relu_stack = nn.Sequential(
            ResidualFcBlock(COLUMNS * 2),
            ResidualFcBlock(COLUMNS * 2),
            ResidualFcBlock(COLUMNS * 2),
            ResidualFcBlock(COLUMNS * 2),
            nn.Linear(COLUMNS * 2, 1),
        )
    
    def forward(self, x, ecg):
        x1 = self.basefc(x)

        x2 = self.resnet(ecg)
        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc(x2)

        x = torch.cat([x1, x2], dim=1)
        return self.linear_relu_stack(x)

class ChannelAttention(nn.Module): #通道注意力机制
    def __init__(self, in_planes, scaling=16):#scaling为缩放比例，
                                           # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module): #空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

class CBAMAttention(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(CBAMAttention, self).__init__()
        self.channel_attention = ChannelAttention(channel, scaling=scaling)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

#18/34
class BasicBlock(nn.Module):
    expansion = 1 #每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):#downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)#BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 测试
        self.cbam = CBAMAttention(out_channel)

        self.downsample = downsample

    def forward(self, x):
        identity = x #捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out

#50,101,152
class Bottleneck(nn.Module):
    expansion = 4#4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,#输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):#block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # 使用 Kaiming 初始化
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            # 初始化偏置为常数
            init.constant_(m.bias, 0)