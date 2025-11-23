import sys
import traceback
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class CustomException(Exception):
    """
    Base custom exception that captures the original exception and traceback.
    Usage:
        raise CustomException("Something went wrong", original_exception)
    """

    def __init__(self, message: str, original: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.original = original

        # Prefer current exc_info() traceback; fall back to original.__traceback__ if present
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = exc_tb or (original.__traceback__ if original is not None else None)
        self.trace = "".join(traceback.format_tb(tb)) if tb else None

        logger.debug(
            "Initialized %s: message=%s original=%s trace=%s",
            self.__class__.__name__,
            self.message,
            repr(self.original),
            self.trace,
        )

    def __str__(self) -> str:
        if self.original:
            return f"{self.message} (caused by {repr(self.original)})"
        return self.message

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "original": repr(self.original) if self.original else None,
            "trace": self.trace,
        }


# Specialized exceptions for common ML pipeline stages
class DataIngestionException(CustomException):
    """Raised when an error occurs during data ingestion."""


class DataValidationException(CustomException):
    """Raised when data fails validation checks."""


class ModelTrainingException(CustomException):
    """Raised when model training fails."""


class ModelNotFoundException(CustomException):
    """Raised when a required model file is missing."""
    # ...existing code...
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    try:
        # simulate underlying error and raise custom exception
        raise DataIngestionException("Failed to ingest data", FileNotFoundError("data.csv not found"))
    except CustomException as e:
        # logger.debug inside CustomException.__init__ will show because logging is DEBUG
        print("String:", str(e))
        print("Dict:", e.to_dict())
# ...existing code...