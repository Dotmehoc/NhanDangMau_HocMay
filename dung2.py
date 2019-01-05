"""
HV : Nguyễn Quốc Dũng
MSHV : 1770470
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from math import sqrt
import warnings
from collections import Counter
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
# ================================================================#
# Hàm tính khoảnh cách Euclid
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
# Hàm xóa cột trong list

def rem_col(matrix, column):
    return [row[:column] + row[column+1:] for row in matrix]


# ================================================================#
# Khởi tạo
# Tạo Gaussian Distribution dataset (x,y)
mean_init = [0, 0]
cov_init =  [[1, 0], [0, 1]]
x, y = np.random.multivariate_normal(mean_init, cov_init, 1000).T
dataSet = np.array(list(zip(x, y)))                                             # Bộ dataset
num_clusters = 3                                                                # với k = 3

C_x = np.random.randint(np.min(dataSet), np.max(dataSet), size=num_clusters)    # tọa đô x của centroid ngẫu nhiên
C_y = np.random.randint(np.min(dataSet), np.max(dataSet), size=num_clusters)    # tọa đô y của centroid ngẫu nhiên
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)                             # điểm centroid
##---------------------------------

# Xem giản đồ phân bố ban đầu của tập dữ liệu đầu vào
# fig = plt.figure(figsize=(8, 6))
# fig.canvas.set_window_title('Bộ dataset ban đầu')
# plt.scatter(x, y, c='blue', s=7)
# plt.scatter(C_x, C_y, marker='*', s=170, c='g')
# plt.title('Phân bố dataset ban đầu')
# plt.show()
#---------------------------------------------------------------#
# ------------------ Giai đoạn 1 ---------------------#

# Phân cụm lần đầu cho dataset (node 0)
kMean_lv0 = KMeans(num_clusters, random_state=0)        # gọi giải thuật kMean
result_0 = kMean_lv0.fit(dataSet)                       # kết quả trả ra của kMean
lbl_lv0 = result_0.labels_                              # lấy nhãn từng điểm
ctr_lv0 = result_0.cluster_centers_                     # centroid của cụm

# Tạo bảng dataframe chứa kết quả (data (x,y) + label)
df_1 = pd.DataFrame(dataSet, columns= ['x','y'])
df_1['Lbl_Name'] = kMean_lv0.labels_                    # Gán nhãn cụm cho bảng kết quả
##---------------------------------
lst_Np1 = []
print("Phân cụm lần thứ nhất:")                         # In kết quả phân cụm
for n in range(num_clusters):
    Cluster_group = df_1.loc[lbl_lv0 == n]              # xem data tương ứng của từng cụm trong bảng
    lst_Np1.append(Cluster_group.shape[0])              # lưu danh sách số lượng phần tử của các cụm
    # print(Cluster_group)                              # In chi tiết tọa độ điểm
    print('Cụm {}, '.format(n) +
          'số hàng: {}, '.format(Cluster_group.shape[0]),
          'số cột: {}'.format(Cluster_group.shape[1]))
print('\n')
##---------------------------------

# ----- Xem giản đồ dạng thường sau khi phân cụm lần đầu
# fig = plt.figure(figsize=(8, 6))
# fig.canvas.set_window_title('Gom cụm lần thứ nhất')
# plt.scatter(dataSet[:,0], dataSet[:,1], c=df_1['Lbl_Name'], s=5 )
# plt.scatter(ctr_lv0[:, 0], ctr_lv0[:, 1], marker='*', s=200, c='g')
# plt.show()

# ------ Xem giản đồ dạng colorbar
# ax = df_1.plot.scatter(x='x', y='y', c=df_1['Lbl_Name'], colormap='plasma', s=8,figsize=(8, 6))
# ax.scatter(ctr_lv0[:, 0], ctr_lv0[:, 1], marker='*', s=200, c='darkblue')    # xem vị trí centroid
# ax.text(0.95, 0.95, 'Gom cụm lần thứ nhất ',transform=ax.transAxes, ha ="right")
# plt.show()
#---------------------------------------------------------------#

# Phân cụm lần thứ 2 cho 3 cụm của lần chia đầu tiên , k = 3
lst_lv1 = []
for i in range(num_clusters):
    cluster_data = df_1[df_1['Lbl_Name'] == i].drop(['Lbl_Name'], axis=1)
    kMean_lv1 = KMeans(num_clusters)
    kMean_lv1.fit(cluster_data)
    cluster_data['Lbl_Name2'] = kMean_lv1.labels_+ (i+1)*10
    lst_lv1.append(cluster_data)
result_2 = pd.concat(lst_lv1)
# print(result_2)
##---------------------------------
lst_Np2 =[]
print("Phân cụm lần thứ 2:")
for df in lst_lv1:
    lbl_lv1 = df['Lbl_Name2']
    lbl_loop = df['Lbl_Name2'].unique().tolist()
    for item in lbl_loop:
        Cluster_group = df.loc[item == lbl_lv1]                 # xem data tương ứng của từng cụm trong bảng
        lst_Np2.append(Cluster_group.shape[0])                  # lưu danh sách số lượng phần tử của các cụm
        # print(Cluster_group)                                  # In chi tiết tọa độ điểm
        print('Cụm {}, '.format(item) +
              'số hàng: {}, '.format(Cluster_group.shape[0]),
              'số cột: {}'.format(Cluster_group.shape[1]))
print(lst_Np2)
print('\n')
##---------------------------------
#
# # Xem giản đồ tại node level 2
# for df in lst_lv1:
#     fig = plt.figure(figsize=(8, 6))
#     fig.canvas.set_window_title( 'Gom cụm lần thứ hai ')
#     plt.scatter(dataSet[:, 0], dataSet[:, 1], c=result_2['Lbl_Name2'], s=5)
#     plt.show()
    ###----------------
# for df in lst_lv1:
#     ax = df.plot.scatter(x='x', y='y', c='Lbl_Name2', colormap='plasma', s=8, figsize=(8, 6))
#     ax.text(0.40, 1.02, 'Gom cụm lần thứ hai', transform=ax.transAxes, ha='right', fontsize=13)
#     #ax.scatter(ctr_lv1[:, 0], ctr_lv1[:, 1], marker='*', s=200, c='darkblue')    # xem vị trí centroid
#     plt.show()
# #---------------------------------------------------------------#

# Phân cụm lần 3 cho 9 cụm đã chia được tại lần chia thứ  2
lst_lv2 = []
for i in range(num_clusters):                       # lấy 3 cụm parent của lần chia thứ 2,
    data = lst_lv1[i]                               # mỗi cụm parent có 3 cụm con
    label_set = list(data.Lbl_Name2.unique())
    for item in label_set:
        cluster_data = data[data['Lbl_Name2'] == item].drop(['Lbl_Name2'], axis=1)
        kMean_lv2 = KMeans(n_clusters=3)
        kMean_lv2.fit(cluster_data)
        cluster_data['Lbl_Name3'] = kMean_lv2.labels_ + (item)*10
        lst_lv2.append(cluster_data)                # lst_lv2 là 9 dataframe,
                                                    # mỗi datafram là 9 cluster của lần chia thứ 2
result_final = pd.concat(lst_lv2)
# print(lst_lv2)
# print(result_final)
# ##---------------------------------
lst_lv3 =[]
lst_Np3 = []
print("Phân cụm lần thứ 3:")
for df in lst_lv2:
    lbl_lv2 = df['Lbl_Name3']
    lbl_loop = df['Lbl_Name3'].unique().tolist()
    for item in lbl_loop:
        Cluster_group = df.loc[item == lbl_lv2]     # xem data tương ứng của từng cụm trong bảng
        lst_lv3.append(Cluster_group)
        lst_Np3.append(Cluster_group.shape[0])      # danh sách số lượng phần tử của các cụm
        # print(Cluster_group)
        print('Cụm {}, '.format(item) +
              'số hàng: {}, '.format(Cluster_group.shape[0]),
              'số cột: {}'.format(Cluster_group.shape[1]))
print('\n')
##---------------------------------
# Xem giản đồ tại node level 3
# for df in lst_lv2:
#     # Xem giản đồ với colorbar sau khi phân cụm lần ba
#     ax = df.plot.scatter(x='x', y='y', c='Lbl_Name3', colormap='plasma', s=8, figsize=(8, 6))
#     ax.text(0.40, 1.02, 'Gom cụm lần thứ ba', transform=ax.transAxes, ha='right', fontsize=13)
#     # ax.scatter(ctr_lv0[:, 0], ctr_lv0[:, 1], marker='*', s=200, c='darkblue')    # xem vị trí centroid
#     plt.show()

#=========================================================================================================#

# ------------------ Giai đoạn 2 ---------------------#
#--- Định trị Sp (tập các mẫu với từng node level) ---#
Sp_lv0 = dataSet
Sp_lv1 = lst_lv1
Sp_lv2 = lst_lv2
Sp_lv3 = lst_lv3
print(Sp_lv1)
#---------------------------------#
#--- Định trị Np (số lượng các mẫu với từng node level) ---#
Np_lv0 = len(df_1)
Np_lv1 = lst_Np1
Np_lv2 = lst_Np2
Np_lv3 = lst_Np3
print('số lượng các phần tử: ', Np_lv1)
#---------------------------------#
#--- Định trị Mp (theo bài báo) ---#
# Tính mean của node level 1
centroid_lv1 = []
for df in lst_lv1:
    x_mean = df['x'].mean()
    y_mean = df['y'].mean()
    centroid_lv1.append([x_mean, y_mean])
# print("03 điểm Mean của node level 1:" ,*centroid_lv1, sep='\n')
print('\n')
#---------------------------------#
# Tính mean của node level 2
centroid_lv2 = []
for df in lst_lv2:
    x_mean = df['x'].mean()
    y_mean = df['y'].mean()
    centroid_lv2.append([x_mean, y_mean])
# print("09 điểm Mean của node level 2:" , *centroid_lv2, sep='\n')
print('\n')
#---------------------------------#
# Tính mean của node level 3
centroid_lv3 = []
for df in lst_lv2:
    for item in df:
        x_mean = df['x'].mean()
        y_mean = df['y'].mean()
        centroid_lv3.append([x_mean, y_mean])
# print("27 điểm Mean của node level 3:" , *centroid_lv3, sep='\n')
#---------------------------------#
# Tính Rp (khoảng cách max) của từng cụm
# Định trị Rp node level #
# for i in Sp_lv1:
#     sp_loop = i['Lbl_Name2'].unique().tolist()
#     for ii in sp_loop:
#         distances = dist(Sp_lv1[ii], centroid_lv1)
#         RP_max = np.argmax(distances)
#         print(RP_max)


# ===================================================================#


