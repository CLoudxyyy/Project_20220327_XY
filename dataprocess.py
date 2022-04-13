import librosa
import matplotlib.pyplot as plt
from scipy import signal
import samplerate
import numpy as np
from extract_bispectrum import polycoherence, plot_polycoherence
import os

def band_pass_filter(original_signal, order, fc1,fc2, fs):
    '''
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fc2: 截止频率
    :param fs: 音频采样率
    :return: 滤波后的音频数据
    '''
    b, a = signal.butter(N=order, Wn=[2*fc1/fs,2*fc2/fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal

def wav_fenge(normal_list, abnormal_list, filedir='training/training-a/'):
    '''

    :param normal_list:
    :param abnormal_list:
    :param filedir:
    :return:
    '''
    #1.读取音频
    for file in os.listdir(filedir):
        if file.endswith(".wav"):
            type = ""
            if file in normal_list:
                type = "normal/"
            if file in abnormal_list:
                type = "abnormal/"

            audio_data, fs = librosa.load(filedir+file, sr=None)
            #2.数字滤波
            audio_data = band_pass_filter(audio_data, 2, 25, 400, fs)
            #3.下采样
            down_sample_audio_data = samplerate.resample(audio_data.T, 1000 / fs, converter_type='sinc_best').T
            #4.归一化
            down_sample_audio_data = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))
            #5.切割音频
            total_num = len(down_sample_audio_data) / (2500)  # 计算切割次数
            # print(len(down_sample_audio_data))
            # print(total_num)
            for j in range(int(total_num)):
                numpy2npy(down_sample_audio_data,
                          j*2500,
                          j*2500+2500,
                          "data/"+type + file.split(".")[0]+"_"+str(j+1)+".npy")
                # print(str(j*2500)+"----"+str(j*2500+2500))

            for j in range(int(total_num-1)):
                numpy2npy(down_sample_audio_data,
                          j * 2500 + 1250,
                          j * 2500 + 1250 +2500,
                          "data/" + type + file.split(".")[0] + "_" + str(int(total_num)+ j + 1) + ".npy")
            print(file, "分割已完成！")
                #print(str(j * 2500 + 1250) + "----" + str(j * 2500 + 1250 +2500))
                #plt.vlines(j * 2500 + 1250, -1.2, 1.2, color="green", linestyle='--', linewidth=1.1)

def numpy2npy(down_sample_audio_data, start, end, savename):
    '''

    :param down_sample_audio_data:
    :param start:
    :param end:
    :param savename:
    :return:
    '''
    # ex_audio_data = down_sample_audio_data[:2500]
    ex_audio_data = down_sample_audio_data[start:end]
    freq1, freq2, bi_spectrum = polycoherence(
        ex_audio_data,
        nfft=1024,
        nperseg=256,
        noverlap=100,
        fs=1000,
        norm=None)
    bi_spectrum = np.array(abs(bi_spectrum))  # calculate bi_spectrum
    bi_spectrum = 255 * (bi_spectrum - np.min(bi_spectrum)) / (np.max(bi_spectrum) - np.min(bi_spectrum))
    np.save(savename, bi_spectrum)

    #a = np.load("test.npy")
def get_type(dir="training/training-a/"):
    '''

    :param dir:
    :return:
    '''
    abnormal_list = []
    normal_list = []
    file_abnormal = open(dir+"RECORDS-abnormal", mode="r")
    for i in file_abnormal.read().split("\n"):
        if i:
            abnormal_list.append(i+".wav")
            #print(i)
    file_abnormal.close()

    file_normal = open(dir + "RECORDS-normal", mode="r")
    for i in file_normal.read().split("\n"):
        if i:
            normal_list.append(i + ".wav")
            #print(i)
    file_normal.close()

    print("Completed!")
    return normal_list, abnormal_list


if __name__=="__main__":
    #wav_fenge(file='training/training-a/a0002.wav', type="abnormal")
    # normal_list, abnormal_list = get_type(dir="training/training-a/")
    # wav_fenge(normal_list, abnormal_list, filedir='training/training-a/')
    
    # for i in os.listdir("training/"):

    for i in ["training-a/", "training-b/"]:
        dir = "training/"+i+"/"
        normal_list, abnormal_list = get_type(dir=dir)
        wav_fenge(normal_list, abnormal_list, filedir=dir)
