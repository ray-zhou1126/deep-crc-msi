import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,auc,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

test_results1 = pd.read_csv('xxx')
print(test_results1.columns)
test_results2 = pd.read_csv('xxx')
print(test_results2.columns)
test_results3 = pd.read_csv('xxx')
print(test_results3.columns)
test_results4 = pd.read_csv('xxx')
print(test_results4.columns)

allcases1 = test_results1['caseid'].unique()
print(allcases1)
allcases2 = test_results2['caseid'].unique()
print(allcases2)
allcases3 = test_results3['caseid'].unique()
print(allcases3)
allcases4 = test_results4['caseid'].unique()
print(allcases4)

casename1 = []
caselabels1 = []
msirates1 = []
for case in allcases1:
    tilesresults = test_results1[test_results1['caseid']==case]
    msinum = sum(tilesresults['predicts'])
    tilenum = len(tilesresults['predicts'])
    if tilenum > 0:
        msirate = msinum/tilenum
        caselabel = int(all(tilesresults['truelabels']))

        caselabels1.append(caselabel)
        msirates1.append(msirate)
        casename1.append(case)
    else: 
        pass

casename2 = []
caselabels2 = []
msirates2 = []
for case in allcases2:
    tilesresults = test_results2[test_results2['caseid']==case]
    msinum = sum(tilesresults['predicts'])
    tilenum = len(tilesresults['predicts'])
    if tilenum > 0:
        msirate = msinum/tilenum
        caselabel = int(all(tilesresults['truelabels']))

        caselabels2.append(caselabel)
        msirates2.append(msirate)
        casename2.append(case)
    else: 
        pass

casename3 = []
caselabels3 = []
msirates3 = []
for case in allcases3:
    tilesresults = test_results3[test_results3['caseid']==case]
    msinum = sum(tilesresults['predicts'])
    tilenum = len(tilesresults['predicts'])
    if tilenum > 0:
        msirate = msinum/tilenum
        caselabel = int(all(tilesresults['truelabels']))

        caselabels3.append(caselabel)
        msirates3.append(msirate)
        casename3.append(case)
    else: 
        pass

casename4 = []
caselabels4 = []
msirates4 = []
for case in allcases4:
    tilesresults = test_results4[test_results4['caseid']==case]
    msinum = sum(tilesresults['predicts'])
    tilenum = len(tilesresults['predicts'])
    if tilenum > 0:
        msirate = msinum/tilenum
        caselabel = int(all(tilesresults['truelabels']))

        caselabels4.append(caselabel)
        msirates4.append(msirate)
        casename4.append(case)
    else: 
        pass

y_pred1 = np.array(msirates1)
y_true1 = np.array(caselabels1)
aucc1 = roc_auc_score(y_true1,y_pred1)
print(aucc1)
y_predn1 = (y_pred1>0.5).astype('uint8')
auccc1 = accuracy_score(y_true1,y_predn1)
print(auccc1)


aaa1 = confusion_matrix(y_true1, y_predn1)
print(aaa1)


label1 = y_true1
predict1 = y_pred1

fpr1,tpr1,_ = roc_curve(label1,predict1)
roc_auc1 = auc(fpr1,tpr1)

y_pred2 = np.array(msirates2)
y_true2 = np.array(caselabels2)
aucc2 = roc_auc_score(y_true2,y_pred2)
print(aucc2)
y_predn2 = (y_pred2>0.5).astype('uint8')
auccc2 = accuracy_score(y_true2,y_predn2)
print(auccc2)


aaa2 = confusion_matrix(y_true2, y_predn2)
print(aaa2)


label2 = y_true2
predict2 = y_pred2

fpr2,tpr2,_ = roc_curve(label2,predict2)
roc_auc2 = auc(fpr2,tpr2)

y_pred3 = np.array(msirates3)
y_true3 = np.array(caselabels3)
aucc3 = roc_auc_score(y_true3,y_pred3)
print(aucc3)
y_predn3 = (y_pred3>0.5).astype('uint8')
auccc3 = accuracy_score(y_true3,y_predn3)
print(auccc3)


aaa3 = confusion_matrix(y_true3, y_predn3)
print(aaa3)


label3 = y_true3
predict3 = y_pred3

fpr3,tpr3,_ = roc_curve(label3,predict3)
roc_auc3 = auc(fpr3,tpr3)

y_pred4 = np.array(msirates4)
y_true4 = np.array(caselabels4)
aucc4 = roc_auc_score(y_true4,y_pred4)
print(aucc4)
y_predn4 = (y_pred4>0.5).astype('uint8')
auccc4 = accuracy_score(y_true4,y_predn4)
print(auccc4)


aaa4 = confusion_matrix(y_true4, y_predn4)
print(aaa4)


label4= y_true4
predict4 = y_pred4

fpr4,tpr4,_ = roc_curve(label4,predict4)
roc_auc4 = auc(fpr4,tpr4)

plt.figure()
lw = 2
plt.plot(fpr1, tpr1, color='darkorange',
         lw=lw, label='     VGG19 AUC = %0.2f' % roc_auc1)
plt.plot(fpr2, tpr2, color='mediumpurple',
         lw=lw, label='  resnet18 AUC = %0.2f' % roc_auc2)
plt.plot(fpr3, tpr3, color='red',
         lw=lw, label='  resnet50 AUC = %0.2f' % roc_auc3)
plt.plot(fpr4, tpr4, color='darkgreen',
         lw=lw, label='resnet101 AUC = %0.2f' % roc_auc4)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('TCGA CRC-DX Patient-level ROC Curve')
plt.legend(loc="lower right")
#plt.show()
print('1')
plt.savefig('./CRCROC4.png')
plt.savefig("./CRCROC4.svg", format="svg")
print('1')