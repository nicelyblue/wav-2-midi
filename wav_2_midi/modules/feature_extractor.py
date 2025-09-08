import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, audio_file, n_bins):
        self.y, self.sr = librosa.load(audio_file)
        self.n_bins = n_bins
        if np.isnan(self.y).any() or np.isinf(self.y).any():
            raise ValueError(f"Invalid data found in audio file {audio_file}")

    def extract_cqt(self):
        cqt = np.abs(librosa.cqt(y=self.y, sr=self.sr, n_bins=self.n_bins, bins_per_octave=12, fmin=librosa.note_to_hz('C1')))
        if np.isnan(cqt).any() or np.isinf(cqt).any():
            raise ValueError("Invalid data found when extracting Constant Q Transform")
        return cqt

    def extract_pitch(self):
        pitch, _ = librosa.core.piptrack(y=self.y, sr=self.sr)
        pitch = pitch[pitch > 0].reshape(-1)
        if np.isnan(pitch).any() or np.isinf(pitch).any():
            raise ValueError("Invalid data found when extracting pitch")
        return pitch

    def extract_chroma(self):
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        chroma = chroma.T
        if np.isnan(chroma).any() or np.isinf(chroma).any():
            raise ValueError("Invalid data found when extracting chroma")
        return chroma

    def extract_tempo(self):
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        if np.isnan(tempo).any() or np.isinf(tempo).any():
            raise ValueError("Invalid data found when extracting tempo")
        return tempo

    def extract_note_onsets(self):
        onsets = librosa.onset.onset_detect(y=self.y, sr=self.sr)
        onset_times = librosa.frames_to_time(onsets, sr=self.sr)
        if np.isnan(onset_times).any() or np.isinf(onset_times).any():
            raise ValueError("Invalid data found when extracting note onsets")
        return onset_times

    def extract_velocities(self):
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        normalized_onset_env = onset_env / np.max(onset_env)
        velocities = np.clip(normalized_onset_env * 127, 0, 127).astype(int)
        if np.isnan(velocities).any() or np.isinf(velocities).any():
            raise ValueError("Invalid data found when extracting velocities")
        return velocities
