
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
    from collections import Counter

    init_vocab = [chr(i) for i in range(0,255)]

    pieces = re.findall(r"([a-zA-Z']{2,}|[^a-zA-Z']{2,})",text.casefold()) # This takes 2+ length elements of puntuaction or words
    pieces_count = Counter(pieces)
    del pieces

    tokens : list = []
    rules = {}

    for p in pieces_count.keys(): # This splits the words and puntuaction strings in chars
        tokens.append([p[i] for i in range(0,len(p))])
    #print(tokens)

    for i in range(0,1000): # 1000 new tokens for example

        pair_count = dict()
        for token_word in tokens:
            token_word_freq = pieces_count.get(''.join(token_word))
            for i in range(0,len(token_word)-1):
                key = (token_word[i],token_word[i+1])
                value = pair_count.get(key)
                pair_count[key] =  value + token_word_freq if value is not None else token_word_freq

        max_count_element = max(pair_count, key=lambda key : pair_count.get(key)) #Get the most repeated key in the dictionary
        #print(max_count_element, pair_count.get(max_count_element))

        rule_value = max_count_element[0] + max_count_element[1]
        rules[max_count_element] = rule_value
        #print(rules)

        for token_word in tokens: # Cada palabra en la lista de palabras segmentada en tokens...
            i = 0
            while i < len(token_word) - 1:
                key = (token_word[i],token_word[i+1])
                value = rules.get(key)
                if value is not None:
                    token_word[i] = value
                    del token_word[i+1]
                else:
                    i+=1
    new_vocab = ([token for token_word in tokens for token in token_word])


    #for rule in rules:
    #    print(str(rule) + ": " + rules.get(rule)) 
    #sorted_list = sorted(pair_count,key=lambda key : pair_count.get(key)) # Testing
    #for i in range(1,len(sorted_list)):
    #   print(sorted_list[-i],":",pair_count.get(sorted_list[-i]))

    print(set(init_vocab).union(new_vocab))

    return set(init_vocab).union(new_vocab)