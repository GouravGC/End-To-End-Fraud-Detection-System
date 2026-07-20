class FraudDetectionException(Exception):
    def __init__(self, message: str, errors=None):
        super().__init__(message)
        self.message = message
        self.errors = errors

    def __str__(self) -> str:
        if self.errors is None:
            return self.message
        return f"{self.message}. Errors: {self.errors}"
