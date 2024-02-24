# This file extracts any input audio section label by using all-in-one model
import allin1
import config as CONF

def extractSongSection(file):
    result = allin1.analyze(file,device=CONF.ML_DEVICES)
    #print(result)
    return result.segments