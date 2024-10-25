from enum import Enum


class DataExtensions(Enum):
    TGZ = ".tgz"
    TAR_BZ2 = ".tar.bz2"

    def __str__(self) -> str:
        return self.value
