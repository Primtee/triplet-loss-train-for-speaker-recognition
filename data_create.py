# coding=utf-8
__author__ = 'NXG'
import os, wave
import contextlib
import collections
from math import ceil
from dataprovider.create.data_management import mik_dir

saved_original_voice_path = '/data/validation_clip/'


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        """
        wave file basic info:
            _wave_params:
                nchannels=1,
                sampwidth=2,
                framerate=8000,
                nframes=1088000,
                comptype='NONE',
                compname='not compressed'

        """
        num_channels = wf.getnchannels()
        print('voice channel is:', num_channels)
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        # assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)  #
        pcm_data = wf.readframes(wf.getnframes())  # nframs is: 1088000  read all data one time
        # note: len of pcm_data is 2176000
        print('the voice length is:{} and sample_rate is:{}'.format(len(pcm_data), sample_rate))
        return pcm_data, sample_rate  # return row data & sample rate


def write_wave(write_path, audio, sample_rate):
    print('write path:', (write_path, sample_rate))
    wf = wave.open(write_path, 'wb')  # mik_dir
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(audio)
    wf.close()


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_collect(frame_duration_ms, audio, sample_rate):
    # audio: all the data
    frame_segment = []
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 30ms   30ms / 1000ms  2 <-> s bytes
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0  # sub second
    while offset + n < len(audio):  # if 1s n = 8000*2
        """
            0-8000*2  0.0  1
            8000*2- 8000*2+8000 1, 1
            8000*2+8000- 8000*2+8000+8000 2 1
        """
        frame_segment.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n
    print('collect all frams:', len(frame_segment))
    return frame_segment  # 4533


def vad_check(sample_rate, frame_duration_ms, padding_duration_ms, frames, write_path):
    num_padding_frames = int(padding_duration_ms // frame_duration_ms)  # 3000 /30   100 frame
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    for index_, frame in enumerate(frames):
        ring_buffer.append(frame)
    human_voiced = b''.join([seg.bytes for seg in ring_buffer])
    human_voiced_len = len(human_voiced)
    if human_voiced_len < 16000:  # human voice length less than 0.5s
        ring_buffer.clear()
        return False  # not human voice
    else:
        if human_voiced_len < 16000 * 6:  # human voice length in [0.5s, 1s]
            full_human_voice_length = 16000 * 6
            copy_num = ceil(full_human_voice_length / human_voiced_len)
            for copy_step in range(0, copy_num, 1):
                human_voiced = human_voiced.__add__(human_voiced)  # Modify here
        write_wave(write_path, human_voiced, sample_rate)
        return True


def check(*path):
    audio, sample_rate = read_wave(path[1])  # read the wav format voice data
    frames = frame_collect(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_check(sample_rate, 30, len(frames) * 30, frames, path[2])
    print('segments:', segments)
    return segments


# 语音流是否小于3秒
if __name__ == '__main__':
    path = 'D:/save'
    save_path_root = 'D"/enrance_voice'
    dir_path = os.listdir(path)
    for cur_path in dir_path:
        name = os.listdir(os.path.join(path, cur_path))
        for cur_name in name:
            cur_name = os.path.join(path, cur_path, cur_name)  # 音频文件全路径
            save_name = os.path.join(save_path_root, cur_path, cur_name)
            check(3, cur_name, save_name)

