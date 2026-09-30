# AI Inference Music Detector

A small audio-recognition project that fingerprints recordings with Towhee and finds the closest match in a local SQLite library. The included notebook is `x.ipynb`; the reusable implementation is in `music_recognition.py`.

## Requirements

- Python (use a clean virtual environment)
- FFmpeg installed and available on `PATH`
- Packages listed in `requirements.txt`

On Windows, create and activate a virtual environment, then install the project packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ffmpeg -version
```

If PowerShell blocks activation, use the Python executable inside `.venv` directly or follow your organization's execution policy.

## Run the notebook

Open `x.ipynb` in VS Code with the Jupyter extension and select the environment where the requirements are installed. Run the cells from top to bottom.

`dataset.csv` contains 114,000 rows of track metadata, but no local audio paths or audio files. Metadata alone cannot be used to generate audio fingerprints.

Add reference recordings under `audio/library/`. Supported extensions are WAV, MP3, FLAC, M4A, OGG, and OPUS. Names formatted as `Artist - Track Title.wav` populate the artist and title fields; other names use the filename as the title and `Unknown artist` as the artist. Add a query recording at `audio/query.wav`, or change `QUERY_AUDIO_PATH` in the notebook.

The notebook indexes available reference recordings and displays the top five nearest matches for a query. Similarity scores rank results; they are not calibrated confidence probabilities. Audio embedding requires a working FFmpeg installation and the Towhee model runtime, and the first embedding may take longer while dependencies or model assets initialize.

## Use the Python API

```python
from music_recognition import MusicIndex

index = MusicIndex()  # Creates music_index.sqlite3 beside music_recognition.py
index.index_directory("audio/library")
matches = index.search_file("audio/query.wav", limit=5)
for match in matches:
    print(match["artist"], "-", match["title"], match["similarity"])
```

The index is local and can be rebuilt by indexing the reference directory again. The generated SQLite database is intentionally excluded from version control.
