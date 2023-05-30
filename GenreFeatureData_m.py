import librosa
import math
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow


class GenreFeatureData:

    "Music audio features for genre classification"
    hop_length = None
    genre_list = [
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
    ]

    dir_trainfolder = "./gtzan/_train"
    dir_devfolder = "./gtzan/_validation"
    dir_testfolder = "./gtzan/_test"
    dir_all_files = "./gtzan"

    train_X_preprocessed_data = "./gtzan/data_train_input.npy"
    train_Y_preprocessed_data = "./gtzan/data_train_target.npy"
    dev_X_preprocessed_data = "./gtzan/data_validation_input.npy"
    dev_Y_preprocessed_data = "./gtzan/data_validation_target.npy"
    test_X_preprocessed_data = "./gtzan/data_test_input.npy"
    test_Y_preprocessed_data = "./gtzan/data_test_target.npy"

    train_X = train_Y = None
    dev_X = dev_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.hop_length = 512

        self.timeseries_length_list = []
        self.trainfiles_list = self.path_to_audiofiles(self.dir_trainfolder)
        self.devfiles_list = self.path_to_audiofiles(self.dir_devfolder)
        self.testfiles_list = self.path_to_audiofiles(self.dir_testfolder)

        self.all_files_list = []
        self.all_files_list.extend(self.trainfiles_list)
        self.all_files_list.extend(self.devfiles_list)
        self.all_files_list.extend(self.testfiles_list)

        self.augmentar = False
        self.timeseries_length = 128

    def load_preprocess_data(self):
        print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))

        # Training set
        self.train_X, self.train_Y = self.extract_audio_features(self.trainfiles_list, True)
        with open(self.train_X_preprocessed_data, "wb") as f:
            np.save(f, self.train_X)
        with open(self.train_Y_preprocessed_data, "wb") as f:
            self.train_Y = self.one_hot(self.train_Y)
            np.save(f, self.train_Y)

        # Validation set
        self.dev_X, self.dev_Y = self.extract_audio_features(self.devfiles_list)
        with open(self.dev_X_preprocessed_data, "wb") as f:
            np.save(f, self.dev_X)
        with open(self.dev_Y_preprocessed_data, "wb") as f:
            self.dev_Y = self.one_hot(self.dev_Y)
            np.save(f, self.dev_Y)

        # Test set
        self.test_X, self.test_Y = self.extract_audio_features(self.testfiles_list)
        with open(self.test_X_preprocessed_data, "wb") as f:
            np.save(f, self.test_X)
        with open(self.test_Y_preprocessed_data, "wb") as f:
            self.test_Y = self.one_hot(self.test_Y)
            np.save(f, self.test_Y)

    def load_deserialize_data(self):

        self.train_X = np.load(self.train_X_preprocessed_data)
        self.train_Y = np.load(self.train_Y_preprocessed_data)

        self.dev_X = np.load(self.dev_X_preprocessed_data)
        self.dev_Y = np.load(self.dev_Y_preprocessed_data)

        self.test_X = np.load(self.test_X_preprocessed_data)
        self.test_Y = np.load(self.test_Y_preprocessed_data)

    def extract_audio_features(self, list_of_audiofiles, augment=False):

        # Allocate empty array for feature data
        data = np.zeros(
            (len(list_of_audiofiles), self.timeseries_length, 128), dtype=np.float64
        )

        target = []
        for i, file in enumerate(list_of_audiofiles):

            # Load audio file
            y, sr = librosa.load(file)

            # Augmentation: Pitch shift
            if augment:
                n_steps = np.random.randint(low=-5, high=5)
                y = librosa.effects.pitch_shift(y, sr, n_steps)

            # Calculate spectrogram
            spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, hop_length=self.hop_length
            )
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            # Store spectrogram in data array
            data[i, :, :] = spectrogram_db.T[0:self.timeseries_length, :]

            # Extract target genre from file path
            target_genre = re.findall(r"[a-z]+(?=/)", file)[0]
            target.append(target_genre)

        return data, target

    def path_to_audiofiles(self, dir_folder):
        return [
            os.path.join(dir_folder, f)
            for f in os.listdir(dir_folder)
            if f.endswith(".wav")
        ]

    def plot_audio_features(self, audio_file):
        y, sr = librosa.load(audio_file)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=self.hop_length)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        plt.figure(figsize=(10, 4))
        specshow(spectrogram_db, sr=sr, hop_length=self.hop_length, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.show()

    def one_hot(self, Y):
        label_binarizer = self.get_label_binarizer()
        return label_binarizer.transform(Y)

    def get_label_binarizer(self):
        from sklearn.preprocessing import LabelBinarizer

        label_binarizer = LabelBinarizer()
        label_binarizer.fit(self.genre_list)
        return label_binarizer
