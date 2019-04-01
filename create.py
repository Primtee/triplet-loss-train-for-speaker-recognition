# coding=utf-8
import os
import cv2
import numpy as np
from PIL import Image
from pydub import AudioSegment
from dataprovider.create.data_management import mik_dir
FRAMERATE = 16000  # 16k语音
HAMMING_TIME = 0.025  # 汉明窗  25ms
STEP_TIME = 0.010  # 帧移   10ms
SPEC_THRESH = 4  # 阈值？？？
IMAGE_WIDTH = 300  # width 宽度


def overlap(X, window_size, window_step):
    assert window_size % 2 == 0, "Window size must be even!"
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    ws = int(window_size)
    ss = int(window_step)
    valid = len(X) - ws
    nw = valid // ss
    out = np.ndarray((nw, 1024), dtype=X.dtype)
    for i in range(nw):
        start = i * ss
        stop = start + ws
        tmp = X[start: stop]
        tmp = np.hamming(ws) * tmp
        sig1024 = np.hstack((tmp, np.zeros(1024 - len(tmp))))
        out[i] = np.fft.fft(sig1024)
    return out[:, :512]


def stft(X, fftsize=128, step=65, mean_normalize=True):
    if mean_normalize:
        X -= X.mean()
    X = overlap(X, fftsize, step)
    return X


def pretty_spectrogram(audio_data, log=True, thresh=5,
                       fft_size=512, step_size=64):
    specgram = np.abs(stft(audio_data, fftsize=fft_size, step=step_size))
    if log:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)  # take log
        specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
        specgram += thresh
    else:
        specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold
    return specgram


def read_voice(audio_file, ext='.mp3'):
    if ext.upper() == '.mp3':
        audio = AudioSegment.from_file(audio_file, 'mp3')
    elif ext.upper() == '.flac':
        audio = AudioSegment.from_file(audio_file)
    elif ext.upper() == '.WAV':
        try:
            audio = AudioSegment.from_file(audio_file)[1300:4800]
        except:
            audio = AudioSegment.from_file(audio_file)  # 可能不要去除静音  [1300:4800]
    else:
        raise Exception
    # print('voice read successfully')
    audio = audio.set_frame_rate(FRAMERATE)  # 读进来音频文件全部转成16K
    # print('audio.sample_width:', audio.sample_width)
    if audio.sample_width == 2:
        data = np.frombuffer(audio._data, np.int16)  # 位数表示
    elif audio.sample_width == 4:
        data = np.frombuffer(audio._data, np.int32)
    elif audio.sample_width == 1:  # 信道发生变化
        # print('audio:', audio._data)
        data = np.frombuffer(audio._data, np.uint8)
    else:
        raise Exception
    x = []
    for chn in range(audio.channels):
        x.append(data[chn::audio.channels])  # [::]相当于后面的是步长step
    x = np.array(x).T  # 转置运算
    # print('x is:', x)
    return FRAMERATE, x


# 图片拼接
def good_img(img, path, name):
    # 每30个像素切分一次，转换成数组，好切片,图片不需要存放到服务器
    print('shape of the compute png:', img.shape)
    if img.shape[1] == 100:  # 传过来的是1秒
        new_img = np.zeros(shape=(512, 300, 3))
        for index in range(3):
            new_img[:, index * 100: index * 100 + 100] = img
        # cv2.imwrite('%s/%s.png' % (path, name), new_img)
        return new_img
    else:
        # 传过来的不是预想的1秒，但是可能小于3秒
        # 图像增强
        if img.shape[1] < 300:
            new_img = np.zeros(shape=(512, 300, 3))
            new_img[:, 0: img.shape[1]] = img  # 如果还不够
            remain = 300 - img.shape[1]
            margin_times = remain // img.shape[1]
            if margin_times > 0:
                for cur_m in range(margin_times):
                    new_img[:, img.shape[1] + img.shape[1] * cur_m: img.shape[1] + img.shape[1] * (cur_m + 1)] = img
            tail = remain % img.shape[1]
            if tail > 0:
                new_img[:, new_img.shape[1] - tail: new_img.shape[1]] = img[:, img.shape[1] - tail: img.shape[1]]
            # print('img en res:', new_img)
            print('the png is good in the server......')
            # cv2.imwrite('%s/%s.png' % (path, name), new_img)
            return new_img
        else:
            # cv2.imwrite('%s/%s.png' % (path, name), img)
            return img


# parm:(wav_path,png_path,color)
def voice2image(param):
    fft_size = int(FRAMERATE * HAMMING_TIME)  # 400
    step_size = int(FRAMERATE * STEP_TIME)  # 160
    audio, color, out_dir, is_register = param  # 之前是一个元组，打包进来的
    slice_list = audio.split('/')  # [Evaluate_Test,SA1.WAV]
    assert len(slice_list) >= 2, 'data must have folder'
    frame_rate, wave_data = read_voice(audio, ext=os.path.splitext(audio)[1])
    if len(wave_data.shape) > 1:
        wave_data = np.mean(wave_data, axis=1)
    wav_spectrogram = pretty_spectrogram(wave_data.astype('float64'),
                                         fft_size=fft_size,
                                         step_size=step_size,
                                         log=True,
                                         thresh=SPEC_THRESH)
    wav_spectrogram = wav_spectrogram / np.max(wav_spectrogram) * 255.0
    spect_width = wav_spectrogram.shape[0]
    batch = 1
    if spect_width > IMAGE_WIDTH:
        batch = int(wav_spectrogram.shape[0] / IMAGE_WIDTH)
    for i in range(0, batch * IMAGE_WIDTH, IMAGE_WIDTH):
        slice_spectrogram = wav_spectrogram[i: i + IMAGE_WIDTH, :]
        spectrogram = slice_spectrogram.astype(np.uint8).T
        spectrogram = np.flip(spectrogram, 0)  # 翻转操作,倒排
        # path = os.path.join(out_dir, slice_list[-1].split('.')[0])
        # if not os.path.exists(path):
        #     os.makedirs(path)
        if color:
            new_im = cv2.applyColorMap(spectrogram, cv2.COLORMAP_JET)
            # 统计蓝色的数目：
            new_im = good_img(new_im, path, slice_list[-1].split('.')[0])
        else:
            print('color is bad')
            new_im = Image.fromarray(spectrogram)
        out_dir = out_dir.replace('.wav', '.png')
        cv2.imwrite(out_dir, new_im)
#
# if __name__ == '__main__':
#     path = 'D:/save'  # 音频文件
#     save_path_png = 'D:/png'
#     dir_path = os.listdir(path)
#     for cur_path in dir_path:
#         name = os.listdir(os.path.join(path, cur_path))
#         for cur_name in name:
#             cur_name_ = os.path.join(path, cur_path, cur_name)  # 音频文件全路径
#             if not os.path.exists(os.path.join(save_path_png, cur_path)):
#                 mik_dir(os.path.join(save_path_png, cur_path))
#             save_path = os.path.join(save_path_png, cur_path, cur_name)
#             print(cur_name_, save_path)
#             # audio, color, out_dir, is_register = param
#             voice2image((cur_name_, True, save_path, True))
#             # 生成图片
