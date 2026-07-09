import torch
from typing import Any

def set_seed(seed: int, deterministic: bool = False):
    import os, random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)              # If there is GPU
    torch.cuda.manual_seed_all(seed)          # multi-GPU

    # 3) cuDNN flags (solo si tienes CUDA/cuDNN)
    if deterministic:
        torch.backends.cudnn.deterministic = True   # use determinist kernels
        torch.backends.cudnn.benchmark = False      
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True 
        
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def accuracy_from_logits(logits, y_true):
    logits_sm = torch.softmax(logits,dim=-1)
    preds = torch.argmax(logits_sm,dim=-1)
    #print("preds=",preds)
    #print("y_true=",y_true)
    correct = torch.sum((preds==y_true).int()).item()
    nelems = torch.numel(preds)
    return correct/nelems

def save_checkpoint(model : torch.nn.Module, optimizer : torch.optim.Optimizer, epoch_step : int, path : str, steps_mode: bool = False, tokenization_file_name : str = None, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_type": model.__class__.__name__,
        "model_config": model.get_config(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch_step": int(epoch_step),
        "steps_mode": bool(steps_mode),
        "tokenization_file_name":tokenization_file_name,
        "extra": {} if not extra else extra,
    }
    torch.save(payload, path)
    
def load_checkpoint(path : str, map_location: str="cpu") -> tuple[dict[str, Any], torch.nn.Module]:
    from src import models

    print("Loading checkpoint from:", path)

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    # Checkpoint validation
    if "model_type" not in ckpt or ckpt["model_type"] is None:
        raise ValueError("Checkpoint has no 'model_type' or 'model_type' is None. Cannot reconstruct model automatically.")

    if "model_config" not in ckpt or ckpt["model_config"] is None:
        raise ValueError("Checkpoint has no 'model_config' or 'model_config' is None. Cannot reconstruct model automatically.")

    if "model" not in ckpt or ckpt["model"] is None:
        raise ValueError("Checkpoint has no 'model' state_dict or 'model' is None.")

    model_type = ckpt["model_type"]
    model_config = ckpt["model_config"]
    print(model_config)
    # Does the model exists?
    try:
        model_cls = getattr(models, model_type)
    except AttributeError as e:
        raise AttributeError(f"There is no model class named '{model_type}' in models.py.") from e

    # Are the model attributes correct?
    try:
        model = model_cls(**model_config)
    except TypeError as e:
        raise TypeError(f"The stored config for '{model_type}' does not match its constructor.") from e

    model.load_state_dict(ckpt["model"], strict=True)

    return ckpt, model


def byte_encode(text:str) -> list: # I apply the byte + 1 logic because I want to reserve the number 0 for padding
   encoded_text = text.encode(encoding="utf-8",errors="replace")
   result = [byte + 1 for byte in encoded_text]
   return result

def byte_decode(list_ids:list) -> str: # I apply the x - 1 logic because the byte_encode function is thought for padding
    list_to_decode = [x - 1 for x in list_ids]
    bytes_to_decode = bytes(list_to_decode)
    return bytes_to_decode.decode(encoding="utf-8",errors="replace")


def gen_text(model : torch.nn.Module,
            text_start : str,
            window : int,
            ntokens : int,
            tokenization_name : str = None,
            temperature : float = None, 
            top_k : int  = None, 
            top_p : float = None, 
            device : torch.device = get_device(), 
            preset : str = None):
    """
    Generate text autoregressively from an initial prompt.

    The function encodes the initial text into byte-level token IDs and then
    generates new tokens one by one. At each generation step, the model receives
    the current context, produces logits for all positions, and only the logits
    from the last position are used to sample the next byte token.

    Decoding can be controlled with temperature, top-k, top-p, or one of the
    predefined presets.

    Parameters
    ----------
    model : torch.nn.Module
        Trained language model used for generation. The model is expected to
        return logits with shape [batch_size, sequence_length, vocab_size].

    text_start : str
        Initial prompt used as the starting context for generation.

    window : int
        Maximum context length kept during generation. In this byte-level
        ASCII-oriented setup, this corresponds to the recent context passed
        back into the model at each step.

    ntokens : int
        Number of new tokens to generate.

    tokenization_name : str, default=None
        The name of the JSON file of the wished tokenization.
        If None, a byte_level tokenization will be used for generation.

    temperature : float or None, default=None
        Value used to scale the logits before sampling. Lower values make the
        generation more deterministic; higher values make it more random.
        If None, the selected preset provides the value.

    top_k : int or None, default=None
        If provided, only the top-k most likely tokens are kept before sampling.

    top_p : float or None, default=None
        If provided, nucleus sampling is applied: only the smallest group of
        tokens whose cumulative probability exceeds top_p is kept.

    device : torch.device, default=get_device()
        Device where the input tensor is created and where the model is expected
        to run.

    preset : str or None, default=None
        Generation preset. Supported values are:
        - "default"
        - "short_stable"
        - "creative"
        - "debug_greedy"

        In "debug_greedy" mode, the function uses argmax instead of sampling.

    Returns
    -------
    str
        The initial prompt plus the generated continuation.

    Notes
    -------
    A tokenization_folder wasn't considered on possible parameters. The tokenization_name field it will
    load the tokenization with that name saved in the default tokenizations folder OR a absolute path can be used instead.
    """
    from . import tokenizer
    
    model.eval()
    
    if tokenization_name is not None:
        token_to_id, id_to_token, rules = tokenizer.load_from_JSON(tokenization_name)
        evaluated_tokens_list = tokenizer.encode(text_start,token_to_id,rules)
    else:    
        evaluated_tokens_list = byte_encode(text_start)

    token_list_result : list = evaluated_tokens_list.copy()

    evaluated_tokens_list = evaluated_tokens_list[-window:] #Is useful if the text_start is bigger than the window

    with torch.no_grad():
        for _ in range(0,ntokens):
            t_text_start = torch.tensor([evaluated_tokens_list],device=device)
            pretemperature_logits = model(t_text_start)[0,-1,:]
            pretemperature_logits[0] = float("-inf") #The id-token 0 is reserver for padding and currently the model doesn't implement it.
            if (preset == "default"):
                if top_p is None: top_p = 0.95 
                if temperature is None: temperature = 0.92           
            elif (preset == "short_stable"):
                if top_p is None: top_p = 0.9
                if temperature is None: temperature = 0.9
            elif (preset == "creative"):
                if top_p is None: top_p = 0.95
                if temperature is None: temperature = 1.1
            else: #debug_greedy
                if temperature is None: temperature = 1

            logits = torch.div(pretemperature_logits,temperature)
            if top_k: logits = __apply_top_k(logits,top_k)
            if top_p: logits = __apply_top_p(logits, top_p)

            logits_sm = torch.softmax(logits,dim=-1)
            # I get the value from the model predictions for the last window tokens-ids (or less if they aren't enough) 
            if preset == "debug_greedy":
                logits_sampled = torch.argmax(logits_sm,dim=-1)
            else:
                logits_sampled = torch.multinomial(logits_sm,1)[-1]
            model_logits = logits_sampled.item()
            # I append (extend) the model prediction token ids to final token-id list
            token_list_result.append(model_logits)
            # I took from the final token-id list the last window tokens-ids (or less if they aren't enough) 
            evaluated_tokens_list = token_list_result[-window:]

    if tokenization_name is not None:
        text_result = tokenizer.decode(token_list_result,id_to_token)
    else:
        text_result = byte_decode(token_list_result)
    
    return text_result

def __apply_top_k(logits : torch.Tensor, k : int):
    if k is None or k <= 0 or k >= logits.numel():
        return logits
    
    topk_vals, _= torch.topk(logits, k)

    threshold = topk_vals[-1] 

    masked = logits.clone()
    masked[masked < threshold] = float("-inf")
    return masked

def __apply_top_p(logits : torch.Tensor, p : float): 
    if p is None or p <= 0 or p >= 1:
        return logits

    probs = torch.softmax(logits, dim=-1) 

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    remove = cum_probs > p
    remove[0] = False

    masked = logits.clone()

    masked[sorted_idx[remove]] = float('-inf')

    return masked


def create_run_features(model : torch.nn.Module, path: str, run_date : str,
                        lr : float , batch_size : int, wd : float, tokenization_file_name : str):
    """Generates a features.file in the given path

    Parameters
    ----------
    model : torch.nn.Module
        The model that yo want to save its features.
    path : str
        The wished path for the file creation.
    run_date,lr,batch_size,wd,tokenization_file_name : str,float,int,float
        The possible extra features to record.
    """
    model_dic = vars(model)
    with open(path, 'a') as featuresfile:
        featuresfile.write("## ---- MODEL ----\n")
        for key in model_dic.keys():
            value = model_dic[key]
            if (isinstance(value,bool)):
                continue
            if (isinstance(value,int) or isinstance(value,float) or isinstance(value,str)):
                txt = str(key) + ": " + str(value) + "\n" 
                featuresfile.write(txt)
        featuresfile.write("## ---- TRAINING ----\n")    
        featuresfile.write("date: " +  str(run_date) + "\n")
        featuresfile.write("lr: " +  str(lr) + "\n")
        featuresfile.write("batch_size: " +  str(batch_size) + "\n")
        featuresfile.write("weight_decay: " +  str(wd) + "\n")
        featuresfile.write("tokenization_file_name: " +  str(tokenization_file_name) + "\n")