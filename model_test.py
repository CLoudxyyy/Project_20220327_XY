from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

class Test_Model():
    def __init__(self, modelpath):
        self.modelpath=modelpath
        self.model = load_model(self.modelpath)

    def predict_result(self, filepath, features_type="mfcc"):
        if features_type=="mfcc":
            x_test = np.load(filepath).reshape((1, 256, 256, 1))
        if features_type=="mel_spec":
            x_test = np.load(filepath).reshape((1, 155, 128, 1))


        result = self.model.predict(x_test)[0].tolist()
        j = result.index(max(result))
        if j == 0:
            leibie="normal"
        else:
            leibie = "abnormal"
        return leibie

def test(modelpath, features_type="mfcc", dataset="test"):
    '''

    :param modelpath:需要加载的模型的地址
    :param features_type:特征的类型，默认值为mfcc，其他可选值为 mel_spec
    :param dataset:默认值为test， 其他可选值为 train和  dev
    :return:返回模型的评价指标
    '''

    y_test = []
    file = open('file/'+dataset+'.txt', mode='r', encoding='utf-8-sig').read().split('\n')
    # mfcc 预测
    x_test = []
    for i in file:
        if i:
            wenjianming = i.split('\t')[0]  # 获取 音频 相对地址
            biaoqian = int(i.split('\t')[1])
            if features_type=="mfcc":
               img_np = np.load(wenjianming)
            if features_type=="mel_spec":
               img_np = np/load_model(wenjianming)

            x_test.append(img_np)
            y_test.append(biaoqian)

    model = load_model(modelpath)
    y_pred = []
    for i in x_test:
        if features_type=="mfcc":
            x_test = i.reshape(1, 256, 256, 1)
        if features_type=="mel_spec":
            x_test = i.reshape(1, 155, 128, 1)

        result = model.predict(x_test)[0].tolist()
        result = result.index(max(result))
        y_pred.append(result)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    print("accuracy: " + str(float(count / len(y_pred))))

    # labels_name = [0, 1, 2, 3]
    labels_name = ['normal', 'abnormal']
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵")
    print(cm)

    plot_confusion_matrix(cm, labels_name, "HAR Confusion Matrix")
    plt.show()
    plt.savefig("pic/HAR_Confusion_Matrix.png")

    target_names = ['normal', 'abnormal']
    print(classification_report(y_test, y_pred, target_names=target_names))


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
if __name__=='__main__':
    #1.预测单个音频的种类
    model = Test_Model('model/best_model_cnn.h5')
    result = model.predict_result(filepath="data/abnormal/a0247_9.npy", features_type="mfcc")
    print("识别种类为："+str(result))
    # 2.测试某个模型的评价指标
    # test(modelpath='model/best_model_cnn.h5', features_type="mfcc", dataset="test")