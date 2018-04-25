# -*- coding:utf-8 -*-

import treePlotter as tpr
import trees as tr
# 
# fr = open('lenses.txt')  # 打开样本集文件  
# # strip() 方法用于移除字符串头尾指定的字符（默认为空格）  
# # split()通过指定分隔符对字符串进行切片，如果参数num 有指定值，则仅分隔 num 个子字符串  
# # str.split(str="", num=string.count(str))  
# # readlines()从文件中一行一行地读数据，返回一个列表；读取的行数据包含换行符  
# # 从样本集文件中读取所有的行，用换行符分开，去除每行行首和行末的空格，保存到列表变量lenses中  
# lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 定义样本集的特征集  
# lensesTree = DecisionTree.createTree(lenses, lensesLabels)  # 调用模块DecisionTree的函数createTree对样本集产生决策树  
# print lensesTree
# storeTree(lensesTree, 'DecisionTreeStorage.txt')  # 将决策树保存到文件中  
# 
# inTree = grabTree('DecisionTreeStorage.txt')  # 从文件中加载决策树  
# DecisionTreePlotter.createPlot(inTree)  # 调用模块DecisionTreePlotter的函数createPlot绘制产生的决策树  
# 
# lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 定义样本集的特征集  
# print DecisionTree.classify0(inTree, lensesLabels, ['young', 'hyper', 'no', 'normal'])
# print DecisionTree.classify(inTree, lensesLabels, ['young', 'hyper', 'no', 'reduced'])
if __name__ == '__main__':
    fr=open('lenses.txt')
    #这里有个for循环，代表一行一行的读数据，一行一行的规范化处理
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lenslabels=['age', 'prescript', 'astimagic', 'tearRate']
    lentree=tr.createTree(lenses,lenslabels)
    lentree
    tpr.createPlot(lentree)
