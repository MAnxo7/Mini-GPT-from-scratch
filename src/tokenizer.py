
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
    init_vocab = [chr(i) for i in range(0,256)]
    pieces = set(re.findall(r"([a-zA-Z']{2,}|[^a-zA-Z']{2,})",text)) # This takes 2+ length elements of puntuaction or words
    tokens : list = []
    rules = {}

    for p in pieces: # This splits the words and puntuaction strings in chars
        tokens.append([p[i] for i in range(0,len(p))])
    #print(tokens)

    for i in range(0,300): # 1000 new tokens for example

        pair_count = dict()
        for token in tokens:
            for i in range(0,len(token)-1):
                key = token[i] + '+' + token[i+1]
                value = pair_count.get(key)
                pair_count[key] =  value + 1 if value is not None else 1

        max_count_element = max(pair_count, key=lambda value : pair_count.get(value)) #Get the most repeated key in the dictionary
        #print(max_count_element, pair_count.get(max_count_element))

        rule_value = "".join(str(max_count_element).split('+'))
        rules[max_count_element] = rule_value
        #print(rules)

        for token_word in tokens: # Cada palabra en la lista de palabras segmentada en tokens...
            i = 0
            while i < len(token_word) - 1:
                key = token_word[i] + '+' + token_word[i+1]
                value = rules.get(key)
                if value is not None:
                    token_word[i] = value
                    del token_word[i+1]
                i+=1
    new_vocab = ([token for token_word in tokens for token in token_word])

    print(set(init_vocab).union(new_vocab))
    return set(init_vocab).union(new_vocab)