from enum import StrEnum

class Anchor_Method(StrEnum):
  SIMILARITY = "similarity"
  MEAN = "mean"

class Aggregate_Method(StrEnum):
  MAX = "max"
  AVG = "avg"
  AVG3TOP = "avg3top"
  