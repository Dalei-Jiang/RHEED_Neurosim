# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:44:13 2024

@author: dalei
"""
import os
import torch
from scipy.signal import convolve2d
from PIL import Image
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from cifar import dataset
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.neighbors import LocalOutlierFactor

def convolution(image_array, kernel, mode='reverse'):
    H,W,C = image_array.shape
    channel_1 = image_array[:, :, 0]
    channel_2 = image_array[:, :, 1]
    channel_3 = image_array[:, :, 2]
    channel_1 = convolve2d(channel_1, kernel, mode='same', boundary='symm')
    channel_2 = convolve2d(channel_2, kernel, mode='same', boundary='symm')
    channel_3 = convolve2d(channel_3, kernel, mode='same', boundary='symm')
    convolved_image = np.stack([channel_1,channel_2,channel_3], axis=-1)
    plt.figure(figsize=(H/150,W/150))
    if mode=='reverse':
        plt.imshow(np.max(convolved_image.astype(np.uint8))-convolved_image.astype(np.uint8))
        plt.axis('off')  # 可选，隐藏坐标轴
        plt.show()
        return np.max(convolved_image.astype(np.uint8))-convolved_image.astype(np.uint8)
    else:
        return convolved_image

def model_loading(label_type, model_name):
    dir_path = '.\\model\\GaN'
    model_path = os.path.join(dir_path, label_type, model_name)
    model = torch.load(model_path).cuda()
    model.eval()
    print(f'Model loaded: {model_path}')
    return model

def epoch_counter(arr):
    i = 1
    while arr[i,2] != 0:
        i+=1
    return i    

def Towerimg_conv():
    img_path='.\\resource\\Sample_UMICH.jpg'
    N = int(input('Please input the pre-smooth kernel size:\n'))
    path_input = input('If you do not change the path, the image is Sample_UMICH.jpg.\nDo you have another img?')
    if path_input != '':
        img_path = path_input
    kernel = input('What is your kernel selection?\n')
    if N < 1:
        assert('N should be positive integers!')
    image = Image.open(img_path)
    image_array = np.array(image)
    H,W,C = image_array.shape
    SobelX = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
    SobelY = np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
    Edge   = np.array([[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]])
    Corn   = np.array([[-2.0,-1.0,0.0],[-1.0,0.0,1.0],[0.0,1.0,2.0]])
    RoundEdge =  np.array([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
    SobelXN = np.array([[1.0,1.74,1.0],[0.0,0.0,0.0],[-0.85,-1.79,-0.85]])
    SobelYN = np.array([[1.0,0.0,-0.85],[1.74,0.0,-1.79],[1.0,0.0,-0.85]])
    EdgeN   = np.array([[0.0,-0.85,0.0],[-0.85,3.78,-0.85],[0.0,-0.85,0.0]])
    CornN   = np.array([[-1.79,-0.85,0.0],[-0.85,0.0,1.0],[0.0,1.0,1.74]])
    RoundEdgeN =  np.array([[-0.85,-0.85,-0.85],[-0.85,8.45,-0.85],[-0.85,-0.85,-0.85]])
    Smooth = np.ones((N,N)) / (N**2)
    match kernel:
        case 'SobelX':
            K = SobelX
        case 'SobelY':
            K = SobelY        
        case 'Edge':
            K = Edge   
        case 'Corn':
            K = Corn   
        case 'RoundEdge':
            K = RoundEdge
        case 'SobelXN':
            K = SobelXN              
        case 'SobelYN':
            K = SobelYN  
        case 'EdgeN':
            K = EdgeN
        case 'CornN':
            K = CornN
        case 'RoundEdgeN':
            K = RoundEdgeN           
    image_array = convolution(image_array, Smooth, mode='same')
    convolved_image = convolution(image_array, K)
    image = Image.fromarray(convolved_image)
    image.save(f'.\\Imgs\\{kernel}.jpg')
    
def PCA_visual():
    X_file = input('X file name: ')
    y_file = input('y file name: ')
    Comment_Label = input('Image Comment: ')
    if y_file == '':
        y_file = 'y_GaN_hex'
    data = np.load(f'.\\resource\\{X_file}.npy')
    labels = np.load(f'.\\resource\\{y_file}.npy')
    
    N = data.shape[0]
    M = np.prod(data.shape[1:])
    data_flattened = data.reshape(N, M)
    
    lof = LocalOutlierFactor(n_neighbors=20)
    outlier_flags = lof.fit_predict(data_flattened)
    non_outliers = outlier_flags == 1
    
    data_filtered = data_flattened[non_outliers]
    labels_filtered = labels[non_outliers]
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_filtered)
    
    F = 32
    plt.figure(figsize=(F, F))
    for label in np.unique(labels_filtered):
        plt.scatter(data_pca[labels_filtered == label, 0], data_pca[labels_filtered == label, 1], 
                    label=f'Spotty class {label}', s=F*4, alpha=1.0)
    
    plt.xlabel('PCA Component 1',fontsize=F)
    plt.ylabel('PCA Component 2',fontsize=F)
    plt.title(f'2D PCA Visualization for {Comment_Label} (Outliers Removed)',fontsize=F)
    plt.legend(fontsize=F)
    # plt.savefig(f'.\\Imgs\\2D PCA Visualization for {Comment_Label} (Outliers Removed).png', format='png', dpi=300)

def train_test_curve(test_log, train_log, epoches):
    F = 16
    alpha = 0.9
    font_size = 1.3
    plt.figure(figsize=(F,F*alpha))
    ax1 = plt.gca()
    x = np.arange(epoches-1)
    ax1.plot(x, train_log[:epoches-1,1],'k-', label='Training Loss')
    ax1.plot(x,  test_log[1:epoches,1],'r-', label='Validation Loss')
    ax1.set_xlabel('epoch number', fontsize=F*font_size)
    ax1.set_ylabel('Loss', fontsize=F*font_size)
    ax1.legend(loc='lower left', fontsize=F*font_size)
    
    ax2 = ax1.twinx()
    ax2.plot(x,   test_log[1:epoches,2], 'b-', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=F*font_size)
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right', fontsize=F*font_size)
    
    plt.title('Train / Validation Loss & Accuracy Over Epochs with kernels', fontsize=F*font_size)
    # plt.savefig('.\\Imgs\\curve_nokernel.png', format='png', dpi=300)

def model_hist(model):
    # 创建一个空的列表来存储所有参数
    all_params = []
    
    # 遍历模型的所有参数，并将其存储到列表中
    for param in model.parameters():
        # 将参数展平为一维张量，然后添加到列表中
        flattened_params = param.data.view(-1).cpu().numpy().tolist()
        all_params.extend(flattened_params)
    
    # 将参数转换为 NumPy 数组
    all_params = np.array(all_params)
    
    # 用户指定直方图的上下界和分箱数量
    lower_bound = -1.0  # 修改为你想要的下界
    upper_bound = 1.0  # 修改为你想要的上界
    num_bins = 50       # 修改为你想要的分箱数量
    
    # 绘制参数值的直方图
    plt.hist(all_params, bins=num_bins, range=(lower_bound, upper_bound), density=True, edgecolor='black')
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Model Parameters')
    plt.grid(True)
    # plt.savefig('.\\Imgs\\Hist_final.png', format='png', dpi=300)
    
def model_attention(model):
    # 获取模型分类器部分的第一层参数
    first_layer = model.classifier[0]
    if hasattr(first_layer, 'weight'):  # 处理 QLinear 或 Linear 层
        first_layer_weights = first_layer.weight.data.cpu().numpy()
    else:
        raise AttributeError("The first layer does not have weights.")
    
    # 计算输入的 480 个位置的重要性热值（取绝对值并求和）
    importance_scores = np.abs(first_layer_weights).mean(axis=0)
    
    # 将重要性得分重塑为 24x20 的形状
    importance_scores_reshaped = importance_scores.reshape(24, 20)
    
    # 将上半部分和下半部分分别取平均值，合成一个 (12, 20) 的新矩阵
    upper_half = importance_scores_reshaped[:12, :]
    lower_half = importance_scores_reshaped[12:, :]
    averaged_scores = (upper_half + lower_half) / 2
    
    # 绘制合成后的热值图
    plt.figure(figsize=(10, 6))
    sns.heatmap(averaged_scores, cmap='viridis', annot=False)
    plt.xlabel('Feature Index (reshaped)')
    plt.ylabel('Feature Index (reshaped)')
    plt.title('Averaged Heatmap of Feature Importance for trained model')
    # plt.savefig('.\\Imgs\\Exp_heat.png', format='png', dpi=300)
    
def Confusion_matrix(model):
    dataset_name = 'GaN'
    datachoice = input('What is your dataset label choice? ')
    Label = input('What is the image label? ')
    _, test_loader = dataset.loading(
        datatype=dataset_name,
        batch_size=1,
        label=datachoice,
        num_workers=0,
        data_root=os.path.join(tempfile.gettempdir(), os.path.join('public_dataset', 'pytorch'))
    )
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Generate the 6x6 confusion matrix with labels fixed from 0 to 5
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=np.arange(6))
    print("Raw Confusion Matrix:\n", conf_matrix)  # Debug print
    conf_matrix_normalized = conf_matrix.astype('float')
    row_sums = conf_matrix.sum(axis=1, keepdims=True)  # Keep row dimension for safe division
    conf_matrix_normalized = np.divide(conf_matrix_normalized, row_sums, where=row_sums != 0) * 100  # Convert to percentages
    print("Normalized Confusion Matrix:\n", conf_matrix_normalized)  # Debug print
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(conf_matrix_normalized, cmap='Reds', vmin=0, vmax=100)
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(conf_matrix_normalized):
        ax.text(j, i, f'{val:.2f}%', ha='center', va='center', color='black')
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(np.arange(6))
    ax.set_yticklabels(np.arange(6))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {Label}')
    plt.show()
