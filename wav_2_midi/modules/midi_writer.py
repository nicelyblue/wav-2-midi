import mido
from mido import MidiFile, MidiTrack, Message

class MIDIWriter:
    def __init__(self, decoded_midi_notes, onsets, velocities, tempo=500000, ticks_per_beat=480):
        self.decoded_midi_notes = decoded_midi_notes
        self.onsets = onsets
        self.velocities = velocities
        self.tempo = tempo
        self.ticks_per_beat = ticks_per_beat
        self.midi_file = self._create_midi_file()

    def _create_midi_file(self):
        midi_file = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        midi_file.tracks.append(track)

        track.append(mido.MetaMessage('set_tempo', tempo=self.tempo))

        onsets_ticks = [int(onset * self.ticks_per_beat) for onset in self.onsets]

        prev_tick = 0
        for idx, (note, tick, velocity) in enumerate(zip(self.decoded_midi_notes, onsets_ticks, self.velocities)):
            delta_tick = tick - prev_tick
            track.append(Message('note_on', note=note, velocity=velocity, time=delta_tick))

            if idx < len(self.onsets) - 1:
                duration = onsets_ticks[idx + 1] - tick
            else:
                duration = self.ticks_per_beat

            track.append(Message('note_off', note=note, velocity=0, time=duration))
            prev_tick = tick

        return midi_file

    def save(self, output_file):
        self.midi_file.save(output_file)