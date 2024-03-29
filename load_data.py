import pandas as pd
from extract_audio import extractAudio, extractVideoFromUrl
from os.path import exists
#open file
#remove most columns
def openPd(filePath):
    return pd.read_csv(filePath)

def cleanCols(df):
    cols=['idx', 'main_speaker', 'title', 'details', 'posted', 'num_views', 'duration']
    df = df.drop(columns=cols)
    return df

def downloadVideos(df):
    for idx,url in enumerate(df['url']):
        fileName_mp4 = "data/videos/video" + str(idx) + ".mp4"
        if not exists(fileName_mp4):
            if extractVideoFromUrl(url, fileName= fileName_mp4) == -1:
                df.drop([idx])
                idx-=1
            else:
                file_mp3 = "data/audios/audio"+str(idx)+".wav"
                extractAudio(fileName_mp4,file_mp3)
        else:
            file_mp3 = "data/audios/audio"+str(idx)+".wav"
            extractAudio(fileName_mp4,file_mp3)
            print(fileName_mp4, " exists")

df = openPd("/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/SpeechFeedback/data/tedx_dataset.csv")
df=cleanCols(df)
downloadVideos(df)

#input feats = num_samples x num_feats
#feats would be avg_vokume, volume_range, hand_guestures, eye_content
#latent rep = num_samples x num_output neurons
#X_train_combined = np.hstack((#input feats, #feats ))
#explainer = shap.Explainer(AE() X_train_combined)
#shap_values = explainer.shap_values(X_train_combined)


#encode
'''

'''
