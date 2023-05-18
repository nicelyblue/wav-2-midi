import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, audio_file):
        self.y, self.sr = librosa.load(audio_file)

    def extract_cqt(self):
        cqt = np.abs(librosa.cqt(y=self.y, sr=self.sr, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1')))
        return cqt    

    def extract_pitch(self):
        pitch, _ = librosa.core.piptrack(y=self.y, sr=self.sr)
        pitch = pitch[pitch > 0].reshape(-1)
        return pitch

    def extract_chroma(self):
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        return chroma.T

    def extract_tempo(self):
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        return tempo

    def extract_note_onsets(self):
        onsets = librosa.onset.onset_detect(y=self.y, sr=self.sr)
        onset_times = librosa.frames_to_time(onsets, sr=self.sr)
        return onset_times

    def extract_velocities(self):
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        normalized_onset_env = onset_env / np.max(onset_env)
        velocities = np.clip(normalized_onset_env * 127, 0, 127).astype(int)
        return velocities    
