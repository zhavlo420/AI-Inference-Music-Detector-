from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Callable

import numpy as np
from towhee import ops, pipe

AUDIO_SUFFIXES = {'.flac', '.m4a', '.mp3', '.ogg', '.opus', '.wav'}
DATABASE_PATH = Path(__file__).with_name('music_index.sqlite3')

_AUDIO_PIPE = (
    pipe.input('url')
    .map('url', 'frames', ops.audio_decode.ffmpeg())
    .map('frames', 'emb', ops.towhee.audio_embedding.nnfp())
)
_EMBEDDING_LOCK = threading.Lock()


class EmptyLibraryError(RuntimeError):
    pass


def _unit_vector(value) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim > 1:
        vector = vector.reshape(-1, vector.shape[-1]).mean(axis=0)
    vector = np.squeeze(vector)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f'Expected a non-empty one-dimensional fingerprint, got {vector.shape}')
    if not np.isfinite(vector).all():
        raise ValueError('Fingerprint contains non-finite values')
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError('Fingerprint has zero magnitude')
    return (vector / norm).astype(np.float32)


def embed_audio(audio_path: str | Path) -> np.ndarray:
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Audio file not found: {path}')

    with _EMBEDDING_LOCK:
        output = _AUDIO_PIPE.output('emb')(str(path)).get()
    if output is None or len(output) == 0:
        raise ValueError(f'No fingerprint was produced for {path.name}')
    return _unit_vector(output)


class MusicIndex:
    def __init__(
        self,
        database_path: str | Path = DATABASE_PATH,
        embedder: Callable[[str | Path], np.ndarray] = embed_audio,
    ) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS tracks (
                    track_key TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )

    @staticmethod
    def _file_key(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as audio_file:
            for chunk in iter(lambda: audio_file.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def index_file(
        self,
        audio_path: str | Path,
        title: str | None = None,
        artist: str = 'Unknown artist',
    ) -> dict[str, str]:
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Audio file not found: {path}')

        vector = _unit_vector(self.embedder(path))
        track_key = self._file_key(path)
        track = {
            'track_key': track_key,
            'title': (title or path.stem).strip() or path.stem,
            'artist': artist.strip() or 'Unknown artist',
            'source_name': path.name,
        }
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO tracks (track_key, title, artist, source_name, embedding, dimensions)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_key) DO UPDATE SET
                    title = excluded.title,
                    artist = excluded.artist,
                    source_name = excluded.source_name,
                    embedding = excluded.embedding,
                    dimensions = excluded.dimensions,
                    indexed_at = CURRENT_TIMESTAMP
                ''',
                (
                    track['track_key'],
                    track['title'],
                    track['artist'],
                    track['source_name'],
                    vector.tobytes(),
                    int(vector.size),
                ),
            )
        return track

    def index_directory(self, directory: str | Path) -> list[dict[str, str]]:
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f'Reference audio directory not found: {root}')

        indexed = []
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            pieces = path.stem.split(' - ', 1)
            artist, title = pieces if len(pieces) == 2 else ('Unknown artist', path.stem)
            indexed.append(self.index_file(path, title=title, artist=artist))
        return indexed

    def search_file(self, audio_path: str | Path, limit: int = 5) -> list[dict[str, str | float]]:
        if limit < 1:
            raise ValueError('limit must be at least 1')
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT track_key, title, artist, source_name, embedding, dimensions FROM tracks'
            ).fetchall()
        if not rows:
            raise EmptyLibraryError('The reference library is empty. Index audio tracks first.')

        query = _unit_vector(self.embedder(Path(audio_path).expanduser()))
        matches = []
        for row in rows:
            if int(row['dimensions']) != query.size:
                continue
            reference = np.frombuffer(row['embedding'], dtype=np.float32)
            similarity = float(np.dot(query, reference))
            matches.append(
                {
                    'track_key': row['track_key'],
                    'title': row['title'],
                    'artist': row['artist'],
                    'source_name': row['source_name'],
                    'similarity': max(-1.0, min(1.0, similarity)),
                }
            )
        matches.sort(key=lambda match: match['similarity'], reverse=True)
        return matches[:limit]

    def list_tracks(self) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT title, artist, source_name, indexed_at FROM tracks ORDER BY title, artist'
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute('SELECT COUNT(*) AS total FROM tracks').fetchone()
        return int(row['total'])
