from src.utils import encode, decode

def test():
    txt = "Hello how are you? I'm fine...!"
    txt_encoded = encode(txt)
    txt_decoded = decode(txt_encoded)
    assert txt == txt_decoded