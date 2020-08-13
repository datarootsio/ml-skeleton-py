# from config import settings as s

# print(s.DYNACONF_DATA_RAW)

from config import settings

assert settings.key == "value"
assert settings.number == 789