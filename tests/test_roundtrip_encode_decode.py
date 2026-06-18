from src.utils import byte_encode, byte_decode

def test():
    txt = "Hello how are you? I'm fine...!"
    txt_encoded = byte_encode(txt)
    txt_decoded = byte_decode(txt_encoded)
    assert txt == txt_decoded