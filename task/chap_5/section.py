#import msaf
import librosa
import seaborn as sns
import feature.section as section
from sf_segmenter.segmenter import Segmenter


audio_file = r"/mnt/f/music4all/pop/0zuCdBhH2uJpbqbr.mp3"

# boundaries, labels = msaf.process(audio_file, plot=True,n_jobs=16,labels_id="cnmf")
# print(boundaries)
# print(labels)
#
sections = section.extractSongSection(audio_file)
print(sections)


# segmenter = Segmenter()
#
# # audio
# boundaries, labs = segmenter.proc_audio(audio_file)
# segmenter.plot(outdir='doc/audio')
# print('boundaries:', boundaries)
# print('labs:', labs)
