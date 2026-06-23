from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZATIONS_FOLDER_PATH = PROJECT_ROOT / "tokenizations"

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

def decode(list_ids:list,id_to_token:dict) -> str:
    decoded_text_list = []

    for i in range(0,len(list_ids)):
        decoded_text_list.append(id_to_token[list_ids[i]])

    decoded_text = "".join(decoded_text_list)

    return decoded_text



def create_bpe_tokenization(text:str,new_tokens:int) -> tuple[dict,dict,dict]: # token_to_id, id_to_token, rules
    import re
    from collections import Counter

    final_vocab = [chr(i) for i in range(0,256)]

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
        if len(pair_count) == 0: # There aren't any pieces remaining to merge
            print("All training pieces have been reduced to 1 token in ",i," iterations")
            break

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

    new_vocab = [rule["result"] for rule in rules.values()]

    final_vocab.extend(new_vocab)  

    token_to_id = {}
    id_to_token = {}
    id_cont = 1

    for token in final_vocab:
        token_to_id[token] = id_cont
        id_to_token[id_cont] = token
        id_cont += 1
    return token_to_id, id_to_token, rules


def save_to_JSON(token_to_id : dict, id_to_token : dict, rules : dict, file_name : str = None, folder_path : str = None):
    import json
    import os, datetime

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if file_name is None:
        name =  current_date + "_tokenization.json"
    else:
        name = file_name 

    if folder_path is None: os.makedirs(TOKENIZATIONS_FOLDER_PATH,exist_ok=True)

    path : Path = Path(folder_path) / name if folder_path is not None else TOKENIZATIONS_FOLDER_PATH / name
    
    json_dict = {}
    rules_list = []
    json_dict["name"] = name
    json_dict["creation_date"] = current_date
    json_dict["vocab_size"] = len(token_to_id)
    json_dict["token_to_id"] = token_to_id
    json_dict["id_to_token"] = id_to_token
    for rule in rules:
        aux_dict = {}
        aux_dict["left"] = rule[0]
        aux_dict["right"] = rule[1]
        rule_value = rules.get(rule)
        aux_dict["result"] = rule_value["result"]
        aux_dict["rank"] = rule_value["rank"]
        rules_list.append(aux_dict.copy())
    json_dict["rules"] = rules_list

    with path.open("w", encoding="utf-8") as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)
    
    return path

def load_from_JSON(file_name : str, folder_path : str = None) -> tuple[dict,dict,dict]:
    import json
    rules = {}
    # If file_name is an absolute path, the left part isn't evaluated (path is equivalent to file_name)
    path : Path = Path(folder_path) / file_name if folder_path is not None else TOKENIZATIONS_FOLDER_PATH / file_name
    print("tokenization_file =", path)
    with path.open("r", encoding="utf-8") as f:
        json_dict = json.load(f)
    
    token_to_id = json_dict["token_to_id"]
    # This next is needed because ids are automatically saved as strings in .json, while in my implementation I use integers as keys 
        # "id" is also a build-in function in python so I use "id_" instead
    id_to_token = {int(id_): token for id_, token in json_dict["id_to_token"].items()} 

    rules_list = json_dict["rules"]
    for rule in rules_list:
        rules[(rule["left"],rule["right"])] = {"result":rule["result"],"rank":rule["rank"]}
    
    return token_to_id, id_to_token, rules

