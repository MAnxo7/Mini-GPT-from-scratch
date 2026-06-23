from src import tokenizer

def test():
    txt = "Hello, sir--don't go.\n"

    token_to_id, id_to_token, rules = tokenizer.create_bpe_tokenization(txt,100)

    txt_encoded = tokenizer.encode(txt,token_to_id,rules)
    txt_decoded = tokenizer.decode(txt_encoded,id_to_token)
    assert txt == txt_decoded