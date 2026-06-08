
def encode(text:str,token_to_id:dict,rules:dict) -> list:
    import re

    pieces = re.findall(r"([a-zA-Z']+|[^a-zA-Z']+)",text)
   
    tokens = []

    for p in pieces: # This splits the words and puntuaction strings in chars
        tokens.append([p[i] for i in range(0,len(p))])
    
    for token_pieces in tokens:
        merges_left = True
        while(merges_left and len(token_pieces) > 1):
            merges_left = False
            possible_rules = []
            for i in range(0,len(token_pieces)-1):
                key = (token_pieces[i],token_pieces[i+1])
                value = rules.get(key)
                if value is not None:
                    merges_left = True
                    possible_rules.append((i,value["rank"],value["result"]))
            if possible_rules:
                best_rule = min(possible_rules,key=lambda rule : rule[1]) # Obtains the best rule based in the lowest rank
                token_pieces[best_rule[0]] = best_rule[2]
                del token_pieces[best_rule[0]+1]
    #print(tokens)
    for token_pieces in tokens: #ID transformation
        for i in range(0,len(token_pieces)):
            token_pieces[i] = token_to_id[token_pieces[i]]
    
    id_list = [id_piece for token_pieces in tokens for id_piece in token_pieces]
  
    return id_list

### RECORDATORIO QUE ESTA DUPLICADO EN UTILS, CARA AL FINAL QUITARLO DE UTILS Y DEJARLO AQU
def decode(list_ids:list,id_to_token:dict) -> str:
    
    for i in range(0,len(list_ids)):
        list_ids[i] = id_to_token[list_ids[i]]

    decoded_text = "".join(list_ids)

    return decoded_text


### RECORDATORIO QUE ESTA DUPLICADO EN UTILS, CARA AL FINAL QUITARLO DE UTILS Y DEJARLO AQUI


def create_bpe_tokenization(text:str,new_tokens:int) -> tuple[dict,dict,dict]: # token_to_id, id_to_token, rules
    import re
    from collections import Counter

    init_vocab = [chr(i) for i in range(0,256)]

    pieces = re.findall(r"([a-zA-Z']{2,}|[^a-zA-Z']{2,})",text) # This takes 2+ length elements of puntuaction or words
    pieces_count = Counter(pieces)
    del pieces

    tokens : list = []
    rules = {}

    for p in pieces_count.keys(): # This splits the words and puntuaction strings in chars
        tokens.append([p[i] for i in range(0,len(p))])
    #print(tokens)

    for i in range(0,new_tokens):

        pair_count = dict()
        for token_word in tokens:
            token_word_freq = pieces_count.get(''.join(token_word))
            for j in range(0,len(token_word)-1):
                key = (token_word[j],token_word[j+1])
                value = pair_count.get(key)
                pair_count[key] =  value + token_word_freq if value is not None else token_word_freq

        max_count_element = max(pair_count, key=lambda key : pair_count.get(key)) #Get the most repeated key in the dictionary

        rule_value = {"rank":i+1,"result":max_count_element[0] + max_count_element[1]}

        rules[max_count_element] = rule_value

        for token_word in tokens: # Cada palabra en la lista de palabras segmentada en tokens...
            i = 0
            while i < len(token_word) - 1:
                key = (token_word[i],token_word[i+1])
                value = rules.get(key)
                if value is not None:
                    token_word[i] = value["result"]
                    del token_word[i+1]
                else:
                    i+=1

    new_vocab = ([rule["result"] for rule in rules.values()])

    final_vocab = set(init_vocab).union(new_vocab)

    token_to_id = {}
    id_to_token = {}
    id_cont = 1

    for token in final_vocab:
        token_to_id[token] = id_cont
        id_to_token[id_cont] = token
        id_cont += 1
    return token_to_id, id_to_token, rules