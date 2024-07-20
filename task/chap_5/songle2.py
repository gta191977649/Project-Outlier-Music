import numpy as np
from scipy.signal import stft
from scipy.fftpack import fft
import librosa
import seaborn as sns
import matplotlib.pyplot as plt

class RefraiD:
    def __init__(self, audio_file, sr=16000, frame_length=4096, hop_length=1280):
        self.y, self.sr = librosa.load(audio_file, sr=sr)
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.run()

    def extract_chroma_vector(self):
        # Compute STFT
        f, t, Zxx = stft(self.y, fs=self.sr, window='hann', nperseg=self.frame_length,
                         noverlap=self.frame_length - self.hop_length)

        # Compute power spectrum
        Pxx = np.abs(Zxx) ** 2

        # Define pitch classes and octave range
        pitch_classes = 12
        oct_low, oct_high = 3, 8

        # Initialize chroma vector
        chroma = np.zeros((pitch_classes, Pxx.shape[1]))

        # Compute chroma vector
        for h in range(oct_low, oct_high + 1):
            for c in range(pitch_classes):
                freq = 440 * 2 ** ((h * 12 + c - 69) / 12)
                freq_bin = int(freq * self.frame_length / self.sr)
                chroma[c, :] += Pxx[freq_bin, :]

        # Normalize chroma vector
        #chroma = chroma / np.max(chroma, axis=0)
        chroma = chroma / np.maximum(np.max(chroma, axis=1, keepdims=True), 1e-8)
        return chroma

    def calculate_similarity(self, chroma):
        n_frames = chroma.shape[1]
        similarity = np.zeros((n_frames, n_frames))

        for t in range(n_frames):
            for l in range(t + 1):
                similarity[t, l] = 1 - np.linalg.norm(chroma[:, t] - chroma[:, t - l]) / np.sqrt(12)

        return similarity

    def calculate_r_all(self, similarity):
        t, _ = similarity.shape
        r_all = np.zeros_like(similarity)

        for l in range(t):
            cumsum = 0
            for tau in range(l, t):
                cumsum += similarity[tau, l]
                r_all[tau, l] = cumsum / (tau - l + 1)

        return r_all

    def find_repeated_sections(self, similarity):
        n_frames = similarity.shape[0]
        R_all = np.zeros(n_frames)

        for l in range(1, n_frames):
            R_all[l] = np.mean(similarity[l:, l])

        # Implement automatic threshold selection using Otsu's method
        threshold = self.otsu_threshold(R_all)

        # Find peaks above threshold
        peaks = np.where(R_all > threshold)[0]

        # Search for line segments
        segments = []
        for peak in peaks:
            segment = self.search_line_segment(similarity[:, peak], peak)
            segments.extend(segment)  # Extend instead of append

        return segments

    def otsu_threshold(self, data):
        # Implement Otsu's method for automatic threshold selection
        hist, bin_edges = np.histogram(data, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        return bin_centers[np.argmax(variance)]

    def search_line_segment(self, similarity_slice, lag):
        # Implement line segment search
        threshold = self.otsu_threshold(similarity_slice)
        above_threshold = similarity_slice > threshold

        segments = []
        start = None
        for i, val in enumerate(above_threshold):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i, lag))
                start = None

        if start is not None:
            segments.append((start, len(above_threshold), lag))

        return segments

    def integrate_repeated_sections(self, segments):
        # Implement integration of repeated sections
        groups = []
        for segment in segments:
            added = False
            for group in groups:
                if self.is_similar_segment(segment, group[0]):
                    group.append(segment)
                    added = True
                    break
            if not added:
                groups.append([segment])

        return groups

    def is_similar_segment(self, seg1, seg2, tolerance=5):

        # Check if seg1 and seg2 are tuples containing tuples
        if isinstance(seg1[0], tuple) and isinstance(seg2[0], tuple):
            start_diff = abs(seg1[0][0] - seg2[0][0])
            end_diff = abs(seg1[0][1] - seg2[0][1])
        else:
            start_diff = abs(seg1[0] - seg2[0])
            end_diff = abs(seg1[1] - seg2[1])

        return start_diff <= tolerance and end_diff <= tolerance

    def detect_modulated_repetition(self, chroma):
        n_frames = chroma.shape[1]
        modulated_similarity = np.zeros((12, n_frames, n_frames))

        for tr in range(12):
            shifted_chroma = np.roll(chroma, tr, axis=0)
            for t in range(n_frames):
                for l in range(t + 1):
                    modulated_similarity[tr, t, l] = 1 - np.linalg.norm(
                        shifted_chroma[:, t] - chroma[:, t - l]) / np.sqrt(12)

        return modulated_similarity

    def select_chorus_sections(self, groups):
        # Implement chorus section selection
        possibilities = []
        for group in groups:
            possibility = sum(seg[1] - seg[0] for seg in group) * np.log((group[0][1] - group[0][0]) / 1.4)
            possibilities.append(possibility)

        best_group_index = np.argmax(possibilities)
        return groups[best_group_index]

    def run(self):
        self.chroma = self.extract_chroma_vector()
        self.similarity = self.calculate_similarity(self.chroma)
        self.r_all = self.calculate_r_all(self.similarity)

        self.segments = self.find_repeated_sections(self.r_all)
        self.groups = self.integrate_repeated_sections(self.segments)
        self.modulated_similarity = self.detect_modulated_repetition(self.chroma)

        # Process modulated similarity (simplified)
        for tr in range(12):
            modulated_segments = self.find_repeated_sections(self.modulated_similarity[tr])
            self.groups.extend(self.integrate_repeated_sections(modulated_segments))

        self.chorus_sections = self.select_chorus_sections(self.groups)


    def plot_similarity_heatmap(self,similarity):
        """
        Plot the full similarity matrix as a heatmap using Seaborn with a color scale from 0 to 1.
        """
        # Set up the matplotlib figure
        plt.figure(figsize=(12, 10))

        # Create the heatmap using seaborn
        sns.heatmap(similarity,
                    cmap='viridis',
                    square=True,
                    cbar_kws={'label': 'Similarity'},
                    vmin=0,  # Set the minimum value for the colorscale
                    vmax=1)  # Set the maximum value for the colorscale

        plt.title('Full Similarity Matrix Heatmap')
        plt.xlabel('Lag')
        plt.ylabel('Time')

        # Invert the y-axis to match the original matrix orientation
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.show()

    def convert_to_time(self, segments_or_chorus):
        """
        Convert frame indices to time in seconds.

        :param segments_or_chorus: List of segments or chorus sections
        :return: List of segments or chorus sections in seconds
        """
        time_conversion = self.hop_length / self.sr

        if isinstance(segments_or_chorus[0], list):
            # Handle chorus sections (list of lists)
            return [[(s[0] * time_conversion, s[1] * time_conversion, s[2] * time_conversion) for s in group]
                    for group in segments_or_chorus]
        else:
            # Handle segments (list of tuples)
            return [(s[0] * time_conversion, s[1] * time_conversion, s[2] * time_conversion)
                    for s in segments_or_chorus]

# Usage example
if __name__ == "__main__":
    refraid = RefraiD("/Users/nurupo/Desktop/dev/audio/bic_camera.mp3")
    refraid.plot_similarity_heatmap(refraid.r_all)
    print(refraid.convert_to_time(refraid.segments))