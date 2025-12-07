class CommandRouter:
    __slots__ = ()

    def route(self, command: str) -> str:
        """Route a command and return a response."""
        # Default implementation - can be overridden
        return f"Received command: {command}"
