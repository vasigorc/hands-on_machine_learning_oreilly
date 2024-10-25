from data_utilities.data_extensions import DataExtensions


def test_enum_values():
    assert DataExtensions.TGZ.value == ".tgz"
    assert DataExtensions.TAR_BZ2.value == ".tar.bz2"


def test_str_representation():
    assert str(DataExtensions.TGZ) == ".tgz"
    assert str(DataExtensions.TAR_BZ2) == ".tar.bz2"
