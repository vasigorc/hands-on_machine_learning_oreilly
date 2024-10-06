from data_utilities.data_loader import DataLoader

SPAM_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = SPAM_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = SPAM_ROOT + "20030228_spam.tar.bz2"
DATASET_NAME = "spam"

spam_loader = DataLoader(DATASET_NAME, SPAM_URL)
