from collections.abc import Iterator, Sequence
from queue import Empty, Queue
from types import TracebackType
from typing import Any, SupportsBytes

import sounddevice as sd


class AudioRecorder:
    __slots__ = ("sample_rate", "channels", "block_size", "dtype", "_audio_queue", "_stream")

    def __init__(self, sample_rate: int, block_size: int, channels: int = 1, dtype: str = "int16") -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.dtype = dtype
        self._audio_queue = Queue[bytes]()
        self._stream: sd.RawInputStream | None = None

    def _audio_callback(self, indata: SupportsBytes, *_: Sequence[Any]) -> None:
        self._audio_queue.put(bytes(indata))

    def __enter__(self) -> AudioRecorder:
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback,
        )
        self._stream.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()

        self._audio_queue = Queue[bytes]()

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        try:
            return self._audio_queue.get(timeout=1.0)
        except Empty as e:
            raise StopIteration from e
