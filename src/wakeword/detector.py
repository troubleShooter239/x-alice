from collections.abc import Sequence

from rapidfuzz.fuzz import ratio


class FuzzyWakeWordDetector:
    __slots__ = ("keyword", "similarity")

    def __init__(self, keyword: str, similarity: float) -> None:
        self.keyword = keyword
        self.similarity = similarity

    def is_wake_word(self, words: Sequence[str]) -> bool:
        return any(ratio(self.keyword, word) > self.similarity for word in words)
