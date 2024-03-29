import json
import requests
from bs4 import BeautifulSoup
import os
from os.path import exists
import subprocess
import librosa

import numpy as np
#os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/SpeechFeedback/env/lib/python3.12/ffmpeg"
tedx_url = "https://www.ted.com/talks/steven_allison_earth_s_original_inhabitants_and_their_role_in_combating_climate_change"
def extractVideoFromUrl(url, fileName):
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    try:
        video = soup.find(id="__NEXT_DATA__")
        data=video.string
        player_data=json.loads(data)['props']['pageProps']['videoData']['playerData']

        vid_content=json.loads(player_data)['resources']['h264'][0]['file']
        video = requests.get(vid_content)
        with open(fileName, 'wb') as f:
            f.write(video.content)
    except:
        print("couldnt get video")
        return -1
def extractAudio(file_mp4,file_mp3):

    with open(file_mp3, 'wb') as f:
            subprocess.call(["ffmpeg", "-y", "-i", file_mp4, file_mp3], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
mfcc_vals = []
def audio_mfcc(dir):
    
    for fname in os.listdir(dir):
        fname=os.path.join(dir, fname)
        x, sample_rate = librosa.load(fname, res_type="kaiser_fast")
        mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
        mfcc_vals.append(mfcc)
        print((mfcc_vals[0].shape))
    return mfcc_vals[0].shape

print(audio_mfcc("data/audios"))
#once you get mfcc for each 
#extractVideoFromUrl("https://www.ted.com/talks/canwen_xu_i_am_not_your_asian_stereotype", "")