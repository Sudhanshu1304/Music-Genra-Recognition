import os
import math
import json
import librosa


testdir='D:/User/Desktop/testdata'
test_jason='test_jason.json'


def Save_Data(dir_path,jason_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segment=10,Samp_rate=22050):

    global c
    Duration=30
    Sample_per_task = Samp_rate*Duration

    data = {
        "mfcc": [],
    }

    num_samples_per_seg=int(Sample_per_task/num_segment)
    expected_num_mfcc_vector_per_segment=math.ceil(num_samples_per_seg/hop_length)


    for dirpath, dirnames, filenames in os.walk(dir_path):

        for file in filenames:
            print(file)
            signal, sr = librosa.load(os.path.join(dirpath, file), sr=Samp_rate)

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
                break
            break

    with open(jason_path, "w") as f:
        json.dump(data, f, indent=4)



# Save_Data(testdir,test_jason)

def Prediction():

    import numpy as np
    import json
    from tensorflow import keras

    def Load_Data(Path):
        with open(Path, "r") as fp:
            data = json.loads(fp.read())
            return data

    data = Load_Data(r'C:\Users\SUDHANSHU\PycharmProjects\NLP\test_jason.json')
    X = data['mfcc']
    X = np.array(X)
    X.flatten()
    new_mod = keras.models.load_model(r'C:\Users\SUDHANSHU\PycharmProjects\NLP\model1')

    print(new_mod.predict_classes(X))