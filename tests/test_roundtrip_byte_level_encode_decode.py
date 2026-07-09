from src.utils import byte_encode, byte_decode

def test():
    # Text with different symbols is the same after a byte-encode and a byte-decode.
    txt = "Hello how are you? I'm fine...!"
    txt_encoded = byte_encode(txt)
    txt_decoded = byte_decode(txt_encoded)
    assert txt == txt_decoded