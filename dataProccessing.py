import glob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from dateutil.parser import parse
from autocorrect import Speller

import re

abbreviations = {
    'Dr.': 'Doctor',
    'Mr.': 'Mister',
    'Mrs.': 'Misess',
    'Ms.': 'Misess',
    'Jr.': 'Junior',
    'Sr.': 'Senior',

    'U.S': 'UNITED STATES',
    'U-S': 'UNITED STATES',
    'U_K': 'UNITED KINGDOM',
    'U_S': 'UNITED STATES',
    'U.K': 'UNITED KINGDOM',
    'U.S': 'UNITED STATES',
    'VIETNAM': 'VIET NAM',
    'VIET NAM': 'VIET NAM',
    'U-N': 'NITED NATIONS',
    'U_N': 'NITED NATIONS',
    'U.N': 'NITED NATIONS',
    'UK': 'UNITED KINGDOM',
    'US': 'UNITED STATES',
    'U-K': 'UNITED KINGDOM',
    'mar': 'March',
    'jan': 'January',
    'feb': 'February',
    'apr': 'April',
    'jun': 'June',
    'jul': 'July',
    'dec': 'December',
    'nov': 'November',
    'oct': 'October',
    'sep': 'September',
    'aug': 'August',
}
contractions_dict = {
    "n't": " not",
    "'s": " is",
    "'m": " am",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would"
}


def normalize(text):
    REGEX_1 = r"(([12]\d|30|31|0?[1-9])[/-](0?[1-9]|1[0-2])[/.-](\d{4}|\d{2}))"
    REGEX_2 = r"((0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9])[/.-](\d{4}|\d{2}))"
    REGEX_3 = r"((\d{4}|\d{2})[/-](0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9]))"
    REGEX_4 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]),? (\d{4}|\d{2}))"
    REGEX_5 = r"(([12]\d|30|31|0?[1-9]) (January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    REGEX_6 = r"((\d{4}|\d{2}) ,?(January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]))"
    REGEX_7 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    COMBINATION_REGEX = "(" + REGEX_1 + "|" + REGEX_2 + "|" + REGEX_3 + "|" + \
                        REGEX_4 + "|" + REGEX_5 + "|" + REGEX_6 + ")"

    for key, value in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(re.escape(key)), value, text, flags=re.IGNORECASE)
    all_dates = re.findall(COMBINATION_REGEX, text)

    for s in all_dates:
        new_date = parse(s[0]).strftime("%d-%m-%Y")
        text = text.replace(s[0], new_date)

    return text

def removeRedundancy(terms):
    print(terms)
    unique_terms = set()
    filtered_terms = []
    for term in terms:
        if term in unique_terms:
            print('111',term)
            continue
        filtered_terms.append(term)
        unique_terms.add(term)
    return  filtered_terms
def data_proccessing(text):
    spell = Speller(lang='en')
    # ___________________________DATA PROCCESSING____________________________________
    # ----------------------------get tokens----------------------------------
    date_format = "%Y-%m-%d"
    print("-------------------")
    print("Orginal text :", text)

    tokens = re.sub(r'[^\w\s]', '', text)
    tokens = re.sub(r'\d+', '', tokens)
    # ////////normalize////////////////////////////

    tokens = normalize(tokens)
    print('maaaaap', tokens)
    tokens = word_tokenize(tokens)
    print('TOKENIZE',tokens)

    # ------------------------remove punctuation----------------------------------------------
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    print("remove punctuation:", tokens)
    # ----------------------------convert to lower case and remove spaces--------------------
    tokens = [w.lower() for w in tokens]
    print("lower:", tokens)
    #////////removeRedundancy///////
    tokens=removeRedundancy(tokens)
    print("removeRedundancy:", tokens)
    # -----------------------------REMOVE STOP WORD---------------------------------------
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [token for token in tokens if token != '']
    print("filter stop words:", tokens)

    # ---------------------------------Steeming-------------------------------------------
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    print("steeming tokens:", tokens)
    # -----------------------------Lemmatizing--------------------------------------------
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    print("lemmatizeing tokens:", tokens)
    # /////////////////////////spell//////////////////////////
    tokens = [spell(token) for token in tokens]

    return tokens

