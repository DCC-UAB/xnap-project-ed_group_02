import librosa
import math
import os
import re
import json
import numpy as np
import pandas as pd

class GenreFeatureData:

    "Music audio features for genre classification"
    hop_length = None
    genre_list = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International',
       'Electronic', 'Instrumental']
    
    dir_trainfolder = "./fma/fma_small"
    dir_devfolder = "./fma/fma_small"
    dir_testfolder = "./fma/fma_small"
    dir_all_files = "./fma"

    train_X_preprocessed_data = "./fma/data_train_input.npy"
    train_Y_preprocessed_data = "./fma/data_train_target.npy"
    dev_X_preprocessed_data = "./fma/data_validation_input.npy"
    dev_Y_preprocessed_data = "./fma/data_validation_target.npy"
    test_X_preprocessed_data = "./fma/data_test_input.npy"
    test_Y_preprocessed_data = "./fma/data_test_target.npy"

    train_X = train_Y = None
    dev_X = dev_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.hop_length = 512

        self.timeseries_length_list = []

        # compute minimum timeseries length, slow to compute, caching pre-computed value of 1290
        # self.precompute_min_timeseries_len()
        # print("min(self.timeseries_length_list) ==" + str(min(self.timeseries_length_list)))
        # self.timeseries_length = min(self.timeseries_length_list)
        self.augmentar=False
        self.timeseries_length = (
            128
        )   # sequence length == 128, default fftsize == 2048 & hop == 512 @ SR of 22050
        #  equals 128 overlapped windows that cover approx ~3.065 seconds of audio, which is a bit small!

    def load_preprocess_data(self):
        #print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))
        #obtenir el dataframe
        self.get_dataframe()
        #get list of train files
        with open("./fma/arxius_train.txt","r") as f:
            self.trainfiles_list=json.load(f)

        # Training set
        self.train_X, self.train_Y = self.extract_audio_features(self.trainfiles_list,True)
        with open(self.train_X_preprocessed_data, "wb") as f:#aixo no canvia
            np.save(f, self.train_X)
        with open(self.train_Y_preprocessed_data, "wb") as f:#aixo no canvia
            self.train_Y = self.one_hot(self.train_Y)
            np.save(f, self.train_Y)
        
        #get list of validation files
        with open("./fma/arxius_validation.txt","r") as f:
            self.devfiles_list=json.load(f)
        # Validation set
        self.dev_X, self.dev_Y = self.extract_audio_features(self.devfiles_list)
        with open(self.dev_X_preprocessed_data, "wb") as f:
            np.save(f, self.dev_X)
        with open(self.dev_Y_preprocessed_data, "wb") as f:
            self.dev_Y = self.one_hot(self.dev_Y)
            np.save(f, self.dev_Y)
        
        #get list of test files
        with open("./fma/arxius_test.txt","r") as f:
            self.testfiles_list=json.load(f)
        # Test set
        self.test_X, self.test_Y = self.extract_audio_features(self.testfiles_list)
        with open(self.test_X_preprocessed_data, "wb") as f:
            np.save(f, self.test_X)
        with open(self.test_Y_preprocessed_data, "wb") as f:
            self.test_Y = self.one_hot(self.test_Y)
            np.save(f, self.test_Y)

    def load_deserialize_data(self):#aixo no canvia en res

        self.train_X = np.load(self.train_X_preprocessed_data)
        self.train_Y = np.load(self.train_Y_preprocessed_data)

        self.dev_X = np.load(self.dev_X_preprocessed_data)
        self.dev_Y = np.load(self.dev_Y_preprocessed_data)

        self.test_X = np.load(self.test_X_preprocessed_data)
        self.test_Y = np.load(self.test_Y_preprocessed_data)

    def precompute_min_timeseries_len(self):
        for file in self.all_files_list:
            print("Loading " + str(file))
            y, sr = librosa.load(file)
            self.timeseries_length_list.append(math.ceil(len(y) / self.hop_length))

    def extract_audio_features(self, list_of_audiofiles,entreno=False):
        target = []
        dataframe=self.df
        self.augmentar=False#en aquest cas no entrar qui mai(esta persidecas)
        if entreno and self.augmentar:
            data = np.zeros(
                (len(list_of_audiofiles)*2, self.timeseries_length, 33), dtype=np.float64
            )
            
            for i, file in enumerate(list_of_audiofiles):
                y, sr = librosa.load("./fma/fma_small/"+file)
                llista_aux=[y]
                RMS=math.sqrt(np.mean(y**2))
                noise=np.random.normal(0, RMS, y.shape[0])
                llista_aux.append(y + noise*0.4)
                
                n=0
                for x in llista_aux:
                    mfcc = librosa.feature.mfcc(
                        y=x, sr=sr, hop_length=self.hop_length, n_mfcc=13
                    )
                    spectral_center = librosa.feature.spectral_centroid(
                        y=x, sr=sr, hop_length=self.hop_length
                    )
                    chroma = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=self.hop_length)
                    spectral_contrast = librosa.feature.spectral_contrast(
                        y=x, sr=sr, hop_length=self.hop_length
                    )
                    id = file.split("/")[1].split(".")[0]
                    id=int(id)
                    genre=self.df.loc[id]
                    target.append(genre)

                    data[i*2+n, :, 0:13] = mfcc.T[0:self.timeseries_length, :]
                    data[i*2+n, :, 13:14] = spectral_center.T[0:self.timeseries_length, :]
                    data[i*2+n, :, 14:26] = chroma.T[0:self.timeseries_length, :]
                    data[i*2+n, :, 26:33] = spectral_contrast.T[0:self.timeseries_length, :]
                    n+=1
                print(
                    "Extracted features audio track %i of %i."
                    % (i + 1, len(list_of_audiofiles))
                )
        else:
            data = np.zeros(
                (len(list_of_audiofiles), self.timeseries_length, 33), dtype=np.float64
            )   
            for i, file in enumerate(list_of_audiofiles):
                y, sr = librosa.load("./fma/fma_small/"+file)
                mfcc = librosa.feature.mfcc(
                    y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13
                )
                spectral_center = librosa.feature.spectral_centroid(
                    y=y, sr=sr, hop_length=self.hop_length
                )
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
                spectral_contrast = librosa.feature.spectral_contrast(
                    y=y, sr=sr, hop_length=self.hop_length
                )

                id = file.split("/")[1].split(".")[0]
                id=int(id)
                genre=dataframe.loc[id]
                genre=genre.genre
                target.append(genre)

                data[i, :, 0:13] = mfcc.T[0:self.timeseries_length, :]
                data[i, :, 13:14] = spectral_center.T[0:self.timeseries_length, :]
                data[i, :, 14:26] = chroma.T[0:self.timeseries_length, :]
                data[i, :, 26:33] = spectral_contrast.T[0:self.timeseries_length, :]

                print(
                    "Extracted features audio track %i of %i."
                    % (i + 1, len(list_of_audiofiles))
                )

        return data, np.expand_dims(np.asarray(target), axis=1)

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    @staticmethod
    def path_to_audiofiles(dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".au"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio
    
    def get_dataframe(self):
        path="./fma/tracks_small.csv"
        df=pd.read_csv(path)
        df.set_index("track_id",inplace=True)
        self.df=df