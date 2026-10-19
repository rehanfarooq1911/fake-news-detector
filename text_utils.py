import re
import string

_PUNCT = re.compile("[%s]" % re.escape(string.punctuation))


def clean_text(text: str) -> str:
    """Text ko saaf karta hai. Training aur prediction dono mein yehi use hota hai."""
    text = str(text)

    # Data leakage hatao: True.csv ki khabrein aksar "CITY (Reuters) - " se shuru hoti hain
    text = re.sub(r"^\s*[A-Z][A-Za-z\.\s,/'-]{0,40}\(Reuters\)\s*-?\s*", "", text)
    text = re.sub(r"\(?reuters\)?", " ", text, flags=re.I)

    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)               # [brackets]
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # links
    text = re.sub(r"<.*?>+", " ", text)                 # html tags
    text = _PUNCT.sub(" ", text)                        # punctuation
    text = re.sub(r"\w*\d\w*", " ", text)               # numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text
