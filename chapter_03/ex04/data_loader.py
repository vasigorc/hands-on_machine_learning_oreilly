import email.parser
from data_utilities.data_extensions import DataExtensions
from data_utilities.data_loader import DataLoader
import email
import email.policy

SPAM_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = SPAM_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = SPAM_ROOT + "20030228_spam.tar.bz2"
SPAM_DATASET = "spam"
HAM_DATASET = "easy_ham"


def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def load_spam_emails():
    spam_loader = DataLoader(
        dataset_name=SPAM_DATASET,
        url=SPAM_URL,
        extension=DataExtensions.TAR_BZ2,
        levels_up=0,
    )
    if spam_loader.is_dataset_dir_empty():
        spam_loader.download_and_extract_data()
    spam_filenames = [
        f for f in sorted(spam_loader.get_dataset_dir().iterdir()) if len(f.name) > 20
    ]
    return [load_email(filepath) for filepath in spam_filenames]


def load_ham_emails():
    ham_loader = DataLoader(
        dataset_name=HAM_DATASET,
        url=HAM_URL,
        extension=DataExtensions.TAR_BZ2,
        levels_up=0,
    )
    if ham_loader.is_dataset_dir_empty():
        ham_loader.download_and_extract_data()
    ham_filenames = [
        f for f in sorted(ham_loader.get_dataset_dir().iterdir()) if len(f.name) > 20
    ]
    return [load_email(filepath) for filepath in ham_filenames]


print(load_spam_emails()[0].get_content().strip())
print(load_ham_emails()[0].get_content().strip())
