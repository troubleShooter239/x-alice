from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.utils.types import Handler


class Command:
    __slots__ = ("name", "patterns", "handler")

    def __init__(self, name: str, patterns: Sequence[str], handler: Handler) -> None:
        self.name = name
        self.patterns = patterns
        self.handler = handler

    def get_map(self) -> dict[str, Handler]:
        return dict.fromkeys(self.patterns, self.handler)
