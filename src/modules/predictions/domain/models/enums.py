import enum


class DoctorDecision(str, enum.Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    OVERRIDDEN = "OVERRIDDEN"