import email.parser
from enum import Enum
from data_utilities.data_extensions import DataExtensions
from data_utilities.data_loader import DataLoader
import email
import email.policy

SPAM_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = SPAM_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = SPAM_ROOT + "20030228_spam.tar.bz2"
SPAM_DATASET = "spam"
HAM_DATASET = "easy_ham"


class DatasetName(Enum):
    SPAM = "spam"
    HAM = "easy_ham"

    def __str__(self) -> str:
        return self.value

    def url(self) -> str:
        match self:
            case DatasetName.SPAM:
                return SPAM_URL
            case DatasetName.HAM:
                return HAM_URL


def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def load_emails(dataset: DatasetName):
    data_loader = DataLoader(
        dataset_name=dataset.value,
        url=dataset.url(),
        extension=DataExtensions.TAR_BZ2,
        levels_up=0,
    )
    if data_loader.is_dataset_dir_empty():
        data_loader.download_and_extract_data()
    filenames = [
        f for f in sorted(data_loader.get_dataset_dir().iterdir()) if len(f.name) > 20
    ]
    return [load_email(filepath) for filepath in filenames]
