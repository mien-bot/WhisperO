# Meeting Mode - Design Document

## Overview
Always-on recording mode for transcribing full meetings with speaker identification.

## Phase 1: Continuous Transcription
Record and transcribe an entire meeting, saving a timestamped transcript file.

### How it works
- Toggle via tray menu: "Start Meeting" / "Stop Meeting"
- Continuously captures audio in ~10-second segments
- Uses Silero VAD (Voice Activity Detection) to skip silence
- Transcribes each speech segment with faster-whisper
- Appends results to a transcript file in real time:
  ```
  ~/.whispero/meetings/Meeting_2026-04-05_14-30.txt
  ```
- Tray icon changes color to indicate meeting mode is active
- "Stop Meeting" finalizes the file and opens it

### Transcript format (Phase 1)
```
Meeting Transcript - 2026-04-05 14:30
======================================

[00:00:12] Let's discuss the Q3 roadmap
[00:00:18] Sure, I think we should focus on the API redesign
[00:01:45] What about the mobile app timeline?
```

### Technical details
- Audio capture: sounddevice InputStream (same as current recording)
- Segmentation: Silero VAD to detect speech/silence boundaries
- Buffer management: process in 10s windows, overlap 0.5s to avoid cut words
- Transcription: same faster-whisper pipeline, runs in background thread
- File I/O: append-mode writes, flush after each segment

---

## Phase 2: Speaker Diarization (Who Said What)
Identify different speakers and label each line of the transcript.

### Approach
Use **pyannote-audio** for speaker diarization alongside faster-whisper for transcription.

### Pipeline
1. Capture audio segment (10s)
2. Run faster-whisper transcription (with word-level timestamps)
3. Run pyannote speaker diarization on the same segment
4. Align whisper timestamps with pyannote speaker labels
5. Merge into labeled transcript

### Transcript format (Phase 2)
```
Meeting Transcript - 2026-04-05 14:30
Speakers: Speaker 1, Speaker 2, Speaker 3
======================================

[00:00:12] Speaker 1: Let's discuss the Q3 roadmap
[00:00:18] Speaker 2: Sure, I think we should focus on the API redesign
[00:01:45] Speaker 3: What about the mobile app timeline?
[00:02:01] Speaker 1: We're targeting end of Q3 for the beta
```

### Speaker labeling options
- Auto-detected as "Speaker 1", "Speaker 2", etc.
- Optional: user can rename speakers during or after the meeting
- Optional: speaker enrollment - record a short voice sample to match names to voices

### Dependencies
- `pyannote-audio` (~1 GB model download, requires HuggingFace token)
- `torch` (already bundled for faster-whisper GPU support)
- `whisperx` (alternative: wraps whisper + pyannote in one pipeline)

### Considerations
- GPU memory: running both whisper and pyannote on GPU needs ~6-8 GB VRAM
- CPU fallback: works but slower (~2-3x real-time on modern CPUs)
- Privacy: all processing is local, no audio leaves the machine
- HuggingFace token: pyannote requires accepting model terms of use

---

## Phase 3: Meeting Intelligence (Future)
- Auto-generate meeting summary (bullet points of key decisions)
- Action item extraction
- Meeting search (search across past transcripts)
- Export to Markdown, PDF, or copy to clipboard
- Integration with calendar (auto-name meetings)

---

## UI Changes
- Tray menu: "Start Meeting" / "Stop Meeting" toggle
- Meeting indicator: tray icon overlay or color change when recording
- Meeting history: "Open Meetings Folder" menu item
- Settings: segment duration, VAD sensitivity, auto-punctuation
