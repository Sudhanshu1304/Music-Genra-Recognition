import os
import math
import json
import librosa

dir = 'D:/User/Desktop/M.L/ML data/Sound/genres_original'
jason_path = "data.json"

'''
n_mfcc=no of feature vectors (13-40)

n_fft=win_size ,

hop_len=stride value

num_segment:
    because we do not have a lot of data to train on so rather then traning model 
    based on the complete audio we will be training it by further dividing it into segments 
'''
c = 0


def Save_Data(dir_path, jason_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segment=8, Samp_rate=22050):
    global c
    Duration = 30
    Sample_per_task = Samp_rate * Duration

    data = {
        "Class": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_seg = int(Sample_per_task / num_segment)
    expected_num_mfcc_vector_per_segment = math.ceil(num_samples_per_seg / hop_length)

    Label = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):

        if dirpath is not dir:  # for leaving the Home Dic name itself

            data["Class"].append((dirpath.split())[-1])  # to store the class name form the path

            for file in filenames:
                print(file)

                try:
                    signal, sr = librosa.load(os.path.join(dirpath, file), sr=Samp_rate)
                    c = c + 1
                except:
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                    continue
                data['labels'].append(Label)
                for s in range(num_segment):

                    start_samp = num_samples_per_seg * s
                    end_samp = start_samp + num_samples_per_seg

                    mfcc = librosa.feature.mfcc(signal[start_samp:end_samp], sr=Samp_rate,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length
                                                )
                    mfcc = mfcc.T
                    if len(mfcc) == expected_num_mfcc_vector_per_segment:
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(Label)

        Label = Label + 1

    with open(jason_path, "w") as f:
        json.dump(data, f, indent=4)


Save_Data(dir, jason_path)

