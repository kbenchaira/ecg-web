import numpy as np
from biosppy.signals.tools import filter_signal


def _bandpass_filter(ecg_signal):
    signal_filtered, _, _ = filter_signal(signal=ecg_signal, ftype='FIR', band='bandpass', order=int(0.3 * 300), frequency=[3, 45],sampling_rate=300)
    return signal_filtered


def SegSig_1d(sig, seg_length=1500, overlap_length=0,full_seg=True, stt=0, seg_num=24):
    length = len(sig)
    SEGs = np.zeros([1, seg_length])
    start = stt
    while start+seg_length <= length:
        tmp = sig[start:start+seg_length].reshape([1, seg_length])
        SEGs = np.concatenate((SEGs, tmp))
        start += seg_length
        start -= overlap_length
    if full_seg:
        if start < length:
            pad_length = seg_length-(length-start)
            tmp = np.concatenate((sig[start:length].reshape([1, length-start]), sig[:pad_length].reshape([1, pad_length])), axis=1)
            SEGs = np.concatenate((SEGs, tmp))

    SEGs = SEGs[1:]
    # print(len(SEGs))
    tmp2 = np.zeros([1, seg_length])
    while len(SEGs) < seg_num:
        # tmp = # code to generate new segments
        SEGs = np.concatenate((SEGs, tmp2))

    return SEGs


def Pad_1d(sig, target_length):
    pad_length = target_length - sig.shape[0]
    if pad_length > 0:
        sig = np.concatenate((sig, np.zeros(int(pad_length))))
    return sig


def Stack_Segs_generate(sig, seg_num=24, seg_length=1500, full_seg=True, stt=0):
    # Filter the signal with bandpass _________________________________________________________________
    sig = _bandpass_filter(sig)
    # padding signals < target_length  ________________________________________________________________
    if len(sig) < seg_length+seg_num:
        sig = Pad_1d(sig, target_length=(seg_length+seg_num-1))
    # calculate the data to be overlapped _____________________________________________________________
    overlap_length = int(seg_length-(len(sig) - seg_length)/(seg_num-1))
    if (len(sig) - seg_length) % (seg_num-1) == 0:
        full_seg = False
    # Slice the signal into segments _________________________________________________________________
    SEGs = SegSig_1d(sig, seg_length=seg_length,overlap_length=overlap_length, full_seg=full_seg, stt=stt, seg_num=seg_num)
    del sig

    SEGs = SEGs.transpose()
    SEGs = SEGs.reshape([1, SEGs.shape[0], SEGs.shape[1]])
    return SEGs


def block_overlap(sig,  seg_num=6, seg_length=2700, full_seg=True, stt=0, buf_size=100):
    SEGt = Stack_Segs_generate( sig, seg_num=seg_num, seg_length=seg_length, full_seg=full_seg, stt=stt)
    return SEGt
