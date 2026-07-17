from collections import Counter
import re


CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
    "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
    "I'm": "I am", "I've": "I have", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will",
    "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
    "might've": "might have", "mightn't": "might not", "must've": "must have",
    "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not",
    "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "so've": "so have", "so's": "so is", "that'd": "that would",
    "that's": "that is", "there'd": "there would", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have",
    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
    "would've": "would have", "wouldn't": "would not", "y'all": "you all",
    "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have",
}

PUNCTS = [
    ",", ".", '"', ":", ")", "(", "-", "!", "?", "|", ";", "'", "$", "&",
    "/", "[", "]", ">", "%", "=", "#", "*", "+", "\\", "•", "~", "@", "£",
    "·", "_", "{", "}", "©", "^", "®", "→", "°", "€", "™", "›", "♥",
    "←", "×", "§", "″", "′", "Â", "█", "½", "à", "…", "–", "●", "â",
    "►", "−", "¢", "²", "░", "¹", "◦", "♦", "ã", "³", "✦", "μ", "ℹ",
    "α", "σ", "⇒", "❌", "👉", "˙", "⚙", "✈", "➡", "😂", "🔥", "📌", "💖",
]


def build_vocab(texts) -> Counter:
    vocab = Counter()
    for text in texts:
        for word in str(text).split():
            vocab[word] += 1
    return vocab


def clean_text(text: str) -> str:
    text = str(text)
    for punct in PUNCTS:
        if punct in text:
            text = text.replace(punct, f" {punct} ")
    return re.sub(r"\s+", " ", text).strip()


def clean_text_v2(text: str) -> str:
    text = str(text)
    text = text.replace("’", "'").replace("‘", "'").replace("´", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"').replace("„", '"')
    text = " ".join(CONTRACTION_MAP.get(word, word) for word in text.split())
    return clean_text(text)
