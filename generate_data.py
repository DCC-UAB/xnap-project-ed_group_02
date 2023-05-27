
def path_to_audiofiles(dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".au"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio

import os
import soundfile as sf
import numpy as np
import math
import io
import re

dir_trainfolder = "./gtzan/_train"

list_of_audiofiles=path_to_audiofiles(dir_trainfolder)

for i, file in enumerate(list_of_audiofiles):


    data, samplerate = sf.read(file)


    RMS=math.sqrt(np.mean(data**2))
    noise=np.random.normal(0, RMS, data.shape[0])
    signal_noise = data+ noise*0.4
    splits = re.split("[ .]", file)
    genre = re.split("[ /]", splits[1])[3]

    sf.write("gtzan_noise/"+genre+".001"+splits[2][3:]+".au",signal_noise,samplerate)

