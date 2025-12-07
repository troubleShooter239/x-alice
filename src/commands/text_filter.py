class TextFilter:
    __slots__ = ("tbr",)

    def __init__(self, tbr: set[str]) -> None:
        self.tbr = tbr

    def filter_tbr(self, text: str) -> str:
        return " ".join(word for word in text.split() if word not in self.tbr)
