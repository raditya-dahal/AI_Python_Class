# import numpy as np
#
# a = np.array([1,2,3,4,5])
# print(a)
# 
# z = np.random.randn(5)
# print(z)
#
# z = np.random.randint(1,6,50)
# print(z)
#
# z = np.random.random(50)
# print(z)
#
# #################################################
#
#
# b = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
# print(b)
# z.reshape(b)
# print(b)
# z.reshape(6,2)
# print(b)
#
#
#
#
#
# ##############################################
#
# c = np.array([[1, 2, 3, 4],
#               [5, 6, 7, 8],
#               [9, 10, 11, 12]])
#
# element = c[1, 2]
# print(element)
# element = c[0,]
# print(element)
# element = c[:, 0]
# print(element)
# element = c[0, :]
# print(element)
#
# ######################################
#
#
# r = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# run = r[0:6:2]
# print(run)
#
# ##########################################
#
# k = np.array([[1,2,3,4,5],
#               [6,7,8,9,10],
#               [11,12,13,14,15],
#               [16,17,18,19,20]])
# print(k)
#
# k[0,0] = k[0,0]*4
# print()
#
# ##############################################
#
# A = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(A)
#
# new = np.vstack((A,A))
# print(new)
# new1 = np.hstack((A,A))
# print(new1)
#
# y1 = np.delete(new1, [0], 1)
# print(y1)
#
# for row in range(0,3):
#     for col in range(0,3):
#         print(new2[row][col])
#
# for i in np.nditer(A):
#     print(i)
#
#
# ###############################################
#
# d = np.array([[14,2,3,4],
#               [15,16,17,20],
#               [19,20,21,22]])
# np.sum(d)
# print(np.sum(d))
# np.mean(d)
# print(np.mean(d))
# np.median(d)
# print(np.median(d))
# np.std(d)
# print(np.std(d))
# mean = np.mean(d)
# st_dev = np.sqrt(np.sum((d - mean) ** 2) / np.size(d))
# print(st_dev)
#
#
# ############################################
# import pandas as pd
#
# import matplotlib.pyplot as plt
#
# x = [450,630,333,459,500]
# y = [2, 6, 8, 10, 12]
# # plt.plot(x,y)
# plt.plot(x,y, color="#576892", marker="d", linestyle=":", linewidth=2)
# plt.xlabel('Price')
# plt.ylabel('Months')
# plt.title('Graph for product price and month')
# plt.show()
#
#
# import pandas as pd
#
# import matplotlib.pyplot as plt
#
# my_File = pd.read_csv('Serial No_GRE Score.csv')
#
# serial_no = my_File['Serial No.']
# GRE = my_File['GRE Score']
#
# plt.plot(serial_no, GRE)
# plt.xlabel('Serial No')
# plt.ylabel('GRE Score')
# plt.title('GRE Scores vs Serial Numbers')
#
# plt.show()
#
#
# ###################################################
#
#
# import pandas as pd
#
# import matplotlib.pyplot as plt
#
# my_File = pd.read_csv('Serial No_GRE Score.csv')
#
# serial_no = my_File['Serial No.']
# GRE = my_File['GRE Score']
#
# plt.bar(serial_no, GRE)
# plt.xlabel('Serial No')
# plt.ylabel('GRE Score')
# plt.title('GRE Scores vs Serial Numbers')
#
# plt.show()
#
#
# ############################################
#
#
# import matplotlib.pyplot as plt
#
# price = [200,300,900,800,100,500]
# profit = [50,30,90,80,10,50]
#
# plt.subplot(3,1,1)
# plt.plot(price,profit)
# plt.subplot(3,1,2)
# plt.plot(price,price)
# plt.title("Balance Sheet")
# plt.subplot(3,1,3)
# plt.plot(price,)
#
#
