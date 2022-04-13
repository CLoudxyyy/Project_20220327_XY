#  制作数据集读取的 文本及标签
import os
import random

def read_file(path, speech_dict):
    train_list =[]
    dev_list = []
    test_list = []
    for i in os.listdir(path):# i  normal、abnormal
        # for k in os.listdir(path+i+'/'): # angry
        npy_list = os.listdir(path+i+'/')
        for num in range(len(npy_list)):  # num 0  1 2 3 4 5 49
            if num <len(npy_list) * 0.7:
                filedir = path+i+'/'+npy_list[num]  # 音频文件相对地址
                leibie=str(speech_dict[i])
                train_list.append(filedir+'\t'+leibie)
            if num < len(npy_list) * 0.85 and num >=len(npy_list) * 0.7:
                filedir = path + i + '/' + npy_list[num]  # 音频文件相对地址
                leibie = str(speech_dict[i])
                train_list.append(filedir + '\t' + leibie)


            if num >= len(npy_list)*0.85:
                filedir = path + i + '/' + npy_list[num]  # 音频文件相对地址
                leibie = str(speech_dict[i])
                train_list.append(filedir + '\t' + leibie)



    # 将训练集 验证集 测试集 打乱顺序
    random.shuffle(train_list)
    random.shuffle(dev_list)
    random.shuffle(test_list)
    return train_list, dev_list, test_list

def file_write(list, type="train"):
    '''
    文本写入
    :param list:
    :return:
    '''
    file = open("file/"+type+".txt", mode='w', encoding='utf-8-sig')

    for i in range(len(list)):
        file.write(list[i]+'\n')

if __name__=='__main__':
    speech_dict = {'normal': 0, 'abnormal': 1}

    train_list, dev_list, test_list = read_file("data/", speech_dict)

    file_write(train_list, type="train")

    file_write(test_list, type="test")

    file_write(dev_list, type="dev")