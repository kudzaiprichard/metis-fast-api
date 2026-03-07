import enum


class Role(str, enum.Enum):
    ADMIN = "ADMIN"
    DOCTOR = "DOCTOR"


class TokenType(str, enum.Enum):
    ACCESS = "ACCESS"
    REFRESH = "REFRESH"