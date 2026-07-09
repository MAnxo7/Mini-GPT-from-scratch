import src.tokenizer as tokenizer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "tiny_shakespare.txt"

def test():
    # The "new_text" section of tiny_shakespare is the same after tokenization,encode and decode of the section
    og_text : str = DATASET_PATH.read_text(encoding="utf-8")
    new_text = og_text[0:int(len(og_text)/4)]

    token_to_id, id_to_token, rules = tokenizer.create_bpe_tokenization(new_text,100)
    list_encode = tokenizer.encode(new_text,token_to_id,rules)
    text_decode = tokenizer.decode(list_encode,id_to_token)

    assert new_text == text_decode

