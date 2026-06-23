import src.tokenizer as tokenizer
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "tiny_shakespare.txt"

def test():

    og_text : str = DATASET_PATH.read_text(encoding="utf-8")
    new_text = og_text[0:int(len(og_text)/4)]
    file_name = "temp_test.json"

    token_to_id, id_to_token, rules = tokenizer.create_bpe_tokenization(new_text,100)
    tokenizer.save_to_JSON(token_to_id,id_to_token,rules,file_name,folder_path=".")
    token_to_id_json, id_to_token_json, rules_json = tokenizer.load_from_JSON(file_name,folder_path=".")
    os.remove(file_name)

    assert token_to_id == token_to_id_json
    assert id_to_token == id_to_token_json
    assert rules == rules_json


