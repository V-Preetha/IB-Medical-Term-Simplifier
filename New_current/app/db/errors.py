"""Typed database error hierarchy.

Callers (services, routes) must never receive raw SQLAlchemy/driver
exceptions. Repository implementations translate driver-level failures into
these typed errors so upstream code can branch on `code`/`status_code`
without importing SQLAlchemy.
"""


class DatabaseError(Exception):
    """Base class for all database-layer errors."""

    code: str = "database_error"
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DatabaseConfigurationError(DatabaseError):
    """Raised when `DatabaseSettings` are missing, invalid, or unsafe."""

    code = "database_configuration_error"
    status_code = 503


class DatabaseUnavailableError(DatabaseError):
    """Raised when the database cannot be reached (connection/pool failure)."""

    code = "database_unavailable"
    status_code = 503


class NotFoundError(DatabaseError):
    """Raised when a requested record does not exist or is not owned by caller."""

    code = "not_found"
    status_code = 404


class ConflictError(DatabaseError):
    """Raised on unique-constraint violations or optimistic-concurrency conflicts."""

    code = "conflict"
    status_code = 409


class TransactionRollbackError(DatabaseError):
    """Raised when a transaction fails and is rolled back."""

    code = "transaction_rollback"
    status_code = 500
