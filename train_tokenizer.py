import argparse
from src import tokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--num-merges", type=int, default=100)
parser.add_argument("--tokenization-name",type=str, default=None)
parser.add_argument("--text-path",type=str, default="tiny_shakespare.txt")
parser.add_argument("--save-dir",type=str, default=None)

args = parser.parse_args()

num_merges = args.num_merges
name = args.tokenization_name
text_path = args.text_path
save_dir = args.save_dir

txt = open(text_path,mode="r").read()

print("Creating tokenization....")
token_to_id, id_to_token, rules = tokenizer.create_bpe_tokenization(txt,num_merges)
file_path = tokenizer.save_to_JSON(token_to_id,id_to_token,rules,name,save_dir)
print("The tokenization has been created correctly: ",file_path)
