import torch.nn as nn
import os
import torch.nn.functional as F
import matplotlib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from keras.utils import to_categorical
from lassonet import LassoNetClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
# from decision_curve import DecisionCurve

kf = KFold(n_splits=5, random_state=None, shuffle=True) # 5折
matplotlib.use('TkAgg')
from sklearn.decomposition import PCA, KernelPCA
from _collections import OrderedDict
import torch
import random, warnings

from lib.pvtv2 import pvt_v2_b0,  pvt_v2_b2
from lib.decoders import EMSCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 训练模型
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 定义RCMFA网络模型
class RCMFA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, l1_lambda, encoder='pvt_v2_b0', pretrain=True, kernel_sizes=[1,3,5],
                 kernel_scale=3, activation='relu', num_classes=1, expansion_factor=2):
        super(RCMFA, self).__init__()
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = './pretrained_pth/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b0.')
            self.backbone = pvt_v2_b2()
            path = './pretrained_pth/pvt/pvt_v2_b0.pth'
            channels=[512, 320, 128, 64]
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        # print('Model %s created, param count: %d' %
        #       (encoder + ' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        #   decoder initialization
        self.decoder = EMSCA(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, kernel_scale=kernel_scale, activation=activation)

        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

        self.fc1 = nn.Linear(192, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.1)
        self.l1_lambda = l1_lambda  # L1 正则化系数
        self.softmax = nn.Softmax(dim=1)  # 添加 softmax 激活函数
    def forward(self, x):

        # 扩展维度到 [178, 1, 189, 189]
        x_expanded = x.unsqueeze(1).unsqueeze(3)  # 变成 [178, 1, 189]
        shape_size_expand=x_expanded.shape[2]
        x = x_expanded.repeat(1, 1, 1, shape_size_expand)
        x = F.interpolate(
            x,
            size=(189, 189),  # 目标大小
            mode='bilinear',  # 双线性插值
            align_corners=False
        )
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        # encoder
        x1, x2, x3, x4 = self.backbone(x)
        # decoder
        dec_outs = self.decoder(x4, [x3, x2, x1])

        # prediction heads
        # p4 = self.out_head4(dec_outs[0])
        # p3 = self.out_head3(dec_outs[1])
        # p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        # p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        # p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        # p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='nearest')

        x_reshaped = p1[:, 0, :, 0]

        x = self.relu(x_reshaped)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x
    def l1_regularization(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

def evaluate(y_true, y_pred_prob, digits=4, cutoff='auto'):
    '''
    calculate several metrics of predictions

    Args:
        y_true: list, labels
        y_pred: list, predictions
        digits: The number of decimals to use when rounding the number. Default is 4
        cutoff: float or 'auto'

    Returns:
        evaluation: dict

    '''
    y_pred = np.argmax(y_pred_prob, axis=1)
    if cutoff == 'auto':
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob[:,1])
        youden = tpr - fpr
        cutoff = thresholds[np.argmax(youden)]
    y_pred_t = [1 if i > cutoff else 0 for i in y_pred_prob[:,1]]
    y_pred_t_ROC = np.array(y_pred_t)
    y_true_ROC = np.array(y_true)
    y_pred_prob_ROC = np.array(y_pred_prob)
    evaluation = OrderedDict()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
    evaluation['auc'] = round(roc_auc_score(y_true, y_pred_t), digits)
    evaluation['auc_ori'] = round(roc_auc_score(y_true, y_pred), digits)
    evaluation['acc'] = round(accuracy_score(y_true, y_pred_t), digits)
    evaluation['recall'] = round(recall_score(y_true, y_pred_t), digits)
    evaluation['specificity'] = round(tn / (tn + fp), digits)
    evaluation['F1'] = round(f1_score(y_true, y_pred_t), digits)
    evaluation['cutoff'] = cutoff
    return evaluation
import jieba
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def preprocess_text(data):
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # 将所有需要进行 LabelEncoder 编码的列转换为字符串类型
    columns_to_encode = [
        'Menopausal status (0.Pre-menopausal; 1. Post-menopausal; 2. Unknown)',
        'Tumor histology(1, Invasive ductal;2, Invasive Lobular; 3, Mixed; 4,Mucinous(粘液）；5， Micropapillary（微乳头）5 大汗腺癌）',
        'HR status', 'HER2', 'Received neoadjuvant chemotherapy',
        '蒽环类Anthracycline', '紫杉烷Taxane', '铂 platinum', 'Anti-HER2（单/双）',
        'Received neoadjuvant endocrine therapy接受新辅助内分泌治疗(来曲唑、阿那曲唑和依西美坦)'
    ]
    # 将这些列转换为字符串
    for column in columns_to_encode:
        data[column] = data[column].astype(str)
    # 将 'Menopausal status' 和 'Tumor histology' 列进行编码
    label_encoder = LabelEncoder()
    # Menopausal status: 0=Pre-menopausal, 1=Post-menopausal, 2=Unknown
    data['Menopausal status'] = label_encoder.fit_transform(
        data['Menopausal status (0.Pre-menopausal; 1. Post-menopausal; 2. Unknown)'])
    # Tumor histology 编码: 1=Invasive ductal, 2=Invasive Lobular, etc.
    data['Tumor histology'] = label_encoder.fit_transform(data[
            'Tumor histology(1, Invasive ductal;2, Invasive Lobular; 3, Mixed; 4,Mucinous(粘液）；5， Micropapillary（微乳头）5 大汗腺癌）'])
    # 其他列根据需求进行编码
    data['HR status'] = label_encoder.fit_transform(data['HR status'])
    data['HER2'] = label_encoder.fit_transform(data['HER2'])
    data['Received neoadjuvant chemotherapy'] = label_encoder.fit_transform(data['Received neoadjuvant chemotherapy'])
    data['蒽环类Anthracycline'] = label_encoder.fit_transform(data['蒽环类Anthracycline'])
    data['紫杉烷Taxane'] = label_encoder.fit_transform(data['紫杉烷Taxane'])
    data['铂 platinum'] = label_encoder.fit_transform(data['铂 platinum'])
    data['Anti-HER2（单/双）'] = label_encoder.fit_transform(data['Anti-HER2（单/双）'])
    data['Received neoadjuvant endocrine therapy接受新辅助内分泌治疗(来曲唑、阿那曲唑和依西美坦)'] = label_encoder.fit_transform(
        data['Received neoadjuvant endocrine therapy接受新辅助内分泌治疗(来曲唑、阿那曲唑和依西美坦)'])
    # 对 'cTNM' 列进行编码，将其作为分类标签
    data['cTNM'] = label_encoder.fit_transform(data['cTNM'])
    data['Clinical stage'] = label_encoder.fit_transform(data['Clinical stage'])
    # 创建一个 SimpleImputer 来填充缺失值（这里选择用列的众数填充分类变量）
    imputer = SimpleImputer(strategy='mean')  # 对分类变量用'mean'填充
    # 填充整个数据集的缺失值
    data = pd.DataFrame(imputer.fit_transform(data.iloc[:, 1:]), columns=data.columns[1:])
    # 将目标列（例如 'label' 或其他）与特征列分离
    data = data.drop(columns=['label'])
    return data
def data_prepare(train_data_value_A,train_data_original,read_clinical_data,batch_Size):
        # 创建一个字典，将标签映射到相应的值
        label_mapping = dict(zip(train_data_value_A['ID'], train_data_value_A['labels']))
        # extracted_All_label = train_data_original['ID'].map(label_mapping)
        clinical_data = preprocess_text(read_clinical_data)
        scaler = StandardScaler()
        clinical_data = scaler.fit_transform(clinical_data)
        pca_clinical_data = PCA(n_components=PCA_num_clinical)  # D维数据保留93
        clinical_data = pca_clinical_data.fit_transform(clinical_data)
        clinical_data = pd.DataFrame(clinical_data)
        # 创建 DataFrame，并插入 ID 列和 `combined` 列
        clinical_data.insert(0, 'ID', read_clinical_data['ID'])
        # train_data_original = clinical_data
        # 按 'ID' 进行分组，并对每一列分别求均值
        train_data_111 = train_data_original.groupby('ID').mean().reset_index()
        clinical_data_111 = clinical_data.groupby('ID').mean().reset_index()
        # 基于 ID 列合并两个 DataFrame
        merged_df = pd.merge(train_data_111, clinical_data_111, on='ID', how='outer')
        merged_df.columns = merged_df.columns.astype(str)
        merged_df_label = merged_df['ID'].map(label_mapping)

        X_train, X_test, y_train, y_test = train_test_split(merged_df.iloc[:, 1:], merged_df_label, test_size=0.2,
                                                           random_state=42, shuffle=True)  # 7 :2
        # 使用SMOTE进行过采样
        smote = SMOTE(sampling_strategy=1)  # 调整到1:1比例
        X_res, y_res = smote.fit_resample(X_train, y_train)
        X_res = scaler.fit_transform(X_res)


        pca_train_data = PCA(n_components=PCA_num)  # D维数据保留
        X_train = pca_train_data.fit_transform(X_res)
        X_test = scaler.fit_transform(X_test)

        pca_test_data = PCA(n_components=100)  # D维数据保留
        X_test = pca_test_data.fit_transform(X_test)
        y_train = y_res
        y_test = y_test
        # # 合并为新的DataFrame
        # train_data_resampled = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=['label'])], axis=1)
        # X_train, X_test, y_train, y_test = train_test_split(train_data, y_res, test_size=0.2,
        #                                                    random_state=42, shuffle=True)  # 7 :2
        ID_num = torch.arange(1, X_train.shape[0] + 1).unsqueeze(1)
        # 对标签进行独热编码
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
        # 转换为 PyTorch Tensor
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        # 训练模型
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, ID_num)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_Size[0], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_Size[1], shuffle=True)
        return train_loader, test_loader, X_test_tensor
'train: val: internal test = 7 : 1 : 2'
train_data_namelist = pd.read_excel(r'.\dataset\classification_2-radiomics_0827.xlsx', header=0, usecols="A:B",
                                   sheet_name="train_label")  # usecols="B:AGF"
# 读取影像组学数据
train_data_original = pd.read_excel(r'.\dataset\classification_2-radiomics_0827.xlsx', header=0,
                                    sheet_name="train_data")  # usecols="B:AGF"
# 读取文本临床数据
read_clinical_data = pd.read_excel(r'.\dataset\classification_2-radiomics_0827.xlsx', header=0,
                   sheet_name="Clinical information_train")  # usecols="B:AGF"
import xgboost as xgb
from catboost import CatBoostClassifier

accuracies = list()
recalls = list()
Precisions = list()
F1_scores = list()
aucs = list()
best_metric = 0
PCA_num = 100  #198  7777max518  189
PCA_num_clinical = 12
best_auc = 0.0
seed = 2
l1_lambda = 0.000000006
# rank = [42,3407,114514,256,208,20010208]
# for l1_lambda in np.arange(6e-05, 1, 0.00005):
for seed in np.arange(2, 3, 1):
    set_seed(seed)
    num_epochs = 20    #30
        #53 47      194   5  3407  372
    hidden_size = 1520
    if PCA_num % 4 != 0:
        input_size = (int(PCA_num/4)+1) * 4
    else:
        input_size = PCA_num
    # 模型初始化
    # input_size = PCA_num  # 输入特征的维度

    output_size = 2  # 2分类问题
    batch_Size = [104,  200]  # 3 130    2 150
    model = RCMFA(input_size, hidden_size, output_size, l1_lambda=l1_lambda)
    # 为CPU中设置种子，生成随机数
    # seed = b
    # 示例数据（替换为你的实际数据）114
    for fold in range(0, 5):# 5
        # c = [num for num in a if num != (b)]  # 其余的数作为c
        print(f"{fold+1}_fold")
        train_loader, test_loader, X_test_tensor = data_prepare(train_data_namelist, train_data_original, read_clinical_data, batch_Size)
        model.__init__(input_size, hidden_size, output_size, l1_lambda)
        model.to(device)
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)#lr=0.0006)
        data = []
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0  # 用于累积每个epoch的损失
            for batch_X, batch_y, ID_num in train_loader:
                optimizer.zero_grad()
                # optimizer1.zero_grad()
                output = model(batch_X.to(device))
                output = output.to('cpu')
                ffnn_loss = criterion(output, batch_y.float())  # 这里不需要再应用 softmax，CrossEntropyLoss 会处理
                # 添加 L1 正则化损失
                # l1_loss = model.l1_regularization()
                # loss = ffnn_loss+l1_loss
                loss = ffnn_loss
                loss.backward()
                optimizer.step()
                # optimizer1.step()
                total_loss += loss.item()
                average_loss = total_loss / len(train_loader)
                # 打印每个epoch的平均损失
                if epoch % 1 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}')
                    batch_y = np.argmax(batch_y, axis=1)
                    # 计算准确率、召回率、F1分数和精确率
                    evaluate_value = evaluate(batch_y, output.data, 4, cutoff='auto')
                    # auc_score = roc_auc_score(batch_y, y_pred, multi_class='ovr')
                    print(f'Training Accuracy: {evaluate_value["acc"] * 100:.2f}%,Training AUC: {evaluate_value["auc"] :.4f}')
        # 在验证集上评估模型
        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for batch_X, batch_y in test_loader:
                output = model(batch_X.to(device))

                output = output.cpu()

                batch_y = np.argmax(batch_y, axis=1)
                # 计算准确率、召回率、F1分数和精确率
                evaluate_val = evaluate(batch_y, output.data, 4, cutoff='auto')
                accuracy = evaluate_val['acc']
                recall = evaluate_val['recall']
                precision = evaluate_val['specificity']
                f1 = evaluate_val['F1']
                # 计算ROC曲线下的面积（AUC）
                # y_pred = np.argmax(output.data, axis=1)
                auc_score = evaluate_val['auc']
                print("train_data shape:", PCA_num)
                # print("feature len:", len_feature)
                print(f"val Accuracy: {evaluate_val['acc'] * 100:.4f}%")
                print(f"val recall: {evaluate_val['recall']* 100:.4f}%")
                print(f"val Precision: {evaluate_val['specificity']* 100:.4f}%")
                print(f"val F1-Score: {evaluate_val['F1']* 100:.4f}%")
                print("val AUC:", evaluate_val['auc'])
                print("val AUC_ori:", evaluate_val['auc_ori'])
                accuracies.append(accuracy)
                recalls.append(recall)
                Precisions.append(precision)
                F1_scores.append(f1)
                aucs.append(auc_score)

        # 模型保存
        metric = auc_score
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            save_dir = 'checkpoints/checkpoint/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = save_dir + str(epoch + 1) + f"best_AUC_{auc_score}_model.pth"
            if auc_score > 0.8:
                torch.save(model.state_dict(), save_path)
                print('saved new best metric model')
    from scipy.stats import norm
    auc_interval = norm.interval(0.95, loc=np.mean(aucs), scale=np.std(aucs) / np.sqrt(len(aucs)))
    auc_interval_rounded = (round(auc_interval[0], 4), round(auc_interval[1], 4))
    accuracy_interval = norm.interval(0.95, loc=np.mean(accuracies), scale=np.std(accuracies) / np.sqrt(len(accuracies)))
    accuracy_interval_rounded = (round(accuracy_interval[0], 4), round(accuracy_interval[1], 4))
    recalls_interval = norm.interval(0.95, loc=np.mean(recalls), scale=np.std(recalls) / np.sqrt(len(recalls)))
    recalls_interval_rounded = (round(recalls_interval[0], 4), round(recalls_interval[1], 4))
    F1_scores_interval = norm.interval(0.95, loc=np.mean(F1_scores), scale=np.std(F1_scores) / np.sqrt(len(F1_scores)))
    F1_scores_interval_rounded = (round(F1_scores_interval[0], 4), round(F1_scores_interval[1], 4))
    Precisions_interval = norm.interval(0.95, loc=np.mean(Precisions), scale=np.std(Precisions) / np.sqrt(len(Precisions)))
    Precisions_interval_rounded = (round(Precisions_interval[0], 4), round(Precisions_interval[1], 4))
    print(f"Average auc: {np.mean(aucs):.4f}±{np.std(aucs) / np.sqrt(len(aucs)):.4f} 95% CI: {auc_interval_rounded}")
    print(f"Average accuracy: {np.mean(accuracies):.4f}±{np.std(accuracies) / np.sqrt(len(accuracies)):.4f} 95% CI: {accuracy_interval_rounded}")
    print(f"Average recalls: {np.mean(recalls):.4f}±{np.std(recalls) / np.sqrt(len(recalls)):.4f} 95% CI: {recalls_interval_rounded}")
    print(f"Average F1_score: {np.mean(F1_scores):.4f}±{np.std(F1_scores) / np.sqrt(len(F1_scores)):.4f}  95% CI: {F1_scores_interval_rounded}")
    print(f"Average Precisions: {np.mean(Precisions):.4f}±{np.std(Precisions) / np.sqrt(len(Precisions)):.4f}  95% CI: {Precisions_interval_rounded}")

    print("seed:", seed)
    if np.mean(aucs) > best_auc:
        best_auc = np.mean(aucs)
        best_metric = seed
    print("Best AUC:", best_auc)
    print("best seed:",best_metric)
    # print("feature len:", len_feature)

# import matplotlib.pyplot as plt
# from draw_shap import draw_shap
# feature_names = pd.read_excel(r'.\dataset\feature_name.xlsx', header=None, nrows=1)#usecols="B:AGF"
# # draw_shap(model=model, feature_names=feature_names, X_test_tensor=X_test_tensor.to(device))
#
# draw_shap(model, X_test_tensor.to(device), feature_names)
#
#

