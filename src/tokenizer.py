
def encode(text:str) -> list:
   text = text.encode(encoding="utf-8",errors="replace")
   return list(text)
### RECORDATORIO QUE ESTA DUPLICADO EN UTILS, CARA AL FINAL QUITARLO DE UTILS Y DEJARLO AQU
def decode(list_bytes:list) -> str:
    bytes_to_decode = bytes(list_bytes)
    return bytes_to_decode.decode(encoding="utf-8",errors="replace")
### RECORDATORIO QUE ESTA DUPLICADO EN UTILS, CARA AL FINAL QUITARLO DE UTILS Y DEJARLO AQUI



def create_bpe_tokenization(text:str) -> None: # De momento ns que devuelve
    import re
    vocab = [i for i in range(0,256)]
    words = set(re.findall(r"[a-zA-Z']{2,}",text))
    puntuaction = set(re.findall("[^a-zA-Z]{2,}",text))
    #puntuaction = re.split(r"a") 
    tokens = []
    for word in words:
        tokens.append([word[i] for i in range(0,len(word))])
    for p in puntuaction:
        tokens.append([p[i] for i in range(0,len(p))])
    print(tokens)
    pair_count = dict()
    for token in tokens:
        for i in range(0,len(token)-1):
            key = token[i] + token[i+1]
            value = pair_count.get(key)
            pair_count[key] =  value + 1 if value is not None else 1
    print(pair_count)