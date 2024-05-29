import re
import os
from dotenv import load_dotenv
from dataclasses import dataclass
import deepl
from googletrans import Translator
from googletrans.models import Translated
from nltk.corpus import wordnet as wn

load_dotenv()
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')


BOOX_TERM_DELIMITER = "--------------------------------------------------"

@dataclass
class Term:
    term: str
    translation_deepl: str
    translation_google: str
    definition: str
    synonims: list[str]

    def __str__(self):
        return f'{self.term} | {self.translation_google}, {self.translation_deepl} | {self.definition}'

    def __repr__(self):
        return self.__str__()

def parse_term(raw_terms: list[str]) -> set[str]:
    pattern = r'^\s{3}(\w+)$' # 3 spaces and then a word
    terms = []
    for term in raw_terms:
        matches = re.findall(pattern, term, re.MULTILINE)
        if matches:
            terms.extend((match.lower() for match in matches))
    return set(terms)


def read_existed_cards() -> set[str]:
    set_of_terms = set()
    with open("data/anki_cards.txt", "r") as f:
        for line in f.readlines():
            term = line.split("|")[0]
            set_of_terms.add(term)
    return set_of_terms


def tranlate_term_deepl(term: str) -> str:
    if not DEEPL_API_KEY:
        raise Exception('No DEEPL_API_KEY provided')
    translator = deepl.Translator(DEEPL_API_KEY)
    result = translator.translate_text(term, source_lang="EN",target_lang="RU")
    if not isinstance(result, list):
        result = [result]
    translation = ', '.join([r.text for r in result])
    return translation

def translate_term_google(term: str) -> str:
    translator = Translator()
    translation = translator.translate(term, src='en', dest='ru')
    if not isinstance(translation, Translated):
        raise Exception('Translation failed for term:', term) 
    return translation.text

def get_definition(term: str) -> str:
    results = []
    synset = wn.synsets(term)[0]
    if synset is not None:
        results.append(synset.definition())
    else:
        results.append('No definition found')
    return '\n'.join(results)

def get_synonyms(term: str) -> list[str]:
    synonyms = []
    for synset in wn.synsets(term):
        for lemma in synset.lemmas(): #type: ignore
            synonyms.append(lemma.name())
    return list(set(synonyms) - {term})[:3]

def fill_term(term: str) -> Term:
    translation_google = translate_term_google(term)
    translation_deepl = tranlate_term_deepl(term)
    definition = get_definition(term)
    synonims = get_synonyms(term)
    return Term(term, translation_deepl, translation_google, definition, synonims)


def reformat_vocabulary(input_file, redo=False) -> list[Term]:
    existed_terms = set()
    if not redo:
        existed_terms = read_existed_cards()
    vocabulary: list[Term] = []
    with open(input_file, 'r') as f:
        content = f.read()
    raw_terms: list[str] = content.split(BOOX_TERM_DELIMITER)
    terms: set[str] = parse_term(raw_terms)
    for term in terms:
        if term not in existed_terms:
            vocabulary.append(fill_term(term))
    return vocabulary


def anki_card_generator(vocabulary: list[Term]) -> None:
    with open("data/anki_cards.txt", "w") as f:
        for term in vocabulary:
            traslation = ",".join({term.translation_google, term.translation_deepl})
            f.write(
                f"{term.term}|({traslation}); {term.definition}; Synonims {term.synonims} \n"
            )

def main():
    input_file = "data/test.txt"
    vocabulary = reformat_vocabulary(input_file, redo=True)
    anki_card_generator(vocabulary)
    print('Anki cards generated successfully')
    print("Check data/anki_cards.txt")
    print(len(vocabulary), 'cards generated')
    print('Vocabulary:', vocabulary)

if __name__ == '__main__':
    main()
