import pretty_midi

def evaluate_transcription(predicted_midi_path, ground_truth_midi_path):
    predicted_midi = pretty_midi.PrettyMIDI(predicted_midi_path)
    ground_truth_midi = pretty_midi.PrettyMIDI(ground_truth_midi_path)

    predicted_notes = [(note.pitch, note.start, note.end) for note in predicted_midi.instruments[0].notes]
    ground_truth_notes = [(note.pitch, note.start, note.end) for note in ground_truth_midi.instruments[0].notes]

    predicted_notes_set = set(predicted_notes)
    ground_truth_notes_set = set(ground_truth_notes)

    true_positives = len(predicted_notes_set & ground_truth_notes_set)
    precision = true_positives / len(predicted_notes) if predicted_notes else 0
    recall = true_positives / len(ground_truth_notes) if ground_truth_notes else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score