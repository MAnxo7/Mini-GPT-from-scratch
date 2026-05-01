import torch

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

def save_checkpoint(model, optimizer, epoch_step, path, steps_mode: bool = False, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch_step": int(epoch_step),
        "steps_mode": bool(steps_mode),
        "extra": extra or {},
    }
    torch.save(payload, path)
    
def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    print(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def encode(text:str) -> list:
   text = text.encode(encoding="utf-8",errors="replace")
   return list(text)

def decode(list_bytes:list) -> str:
    bytes_to_decode = bytes(list_bytes)
    return bytes_to_decode.decode(encoding="utf-8",errors="replace")


def gen_text(model : torch.nn.Module , text_start : str , window : int, ntokens : int ,temperature : float = None, top_k : int  = None, top_p : float = None, device : torch.device = get_device(), preset : str = None):

    model.eval()

    text_result : str = text_start
    byte_text = encode(text_start)

    for _ in range(0,ntokens):
        t_text_start = torch.tensor([byte_text],device=device)
        pretemperature_logits = model(t_text_start)[0,-1,:]
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

        if preset == "debug_greedy":
            logits_sampled = torch.argmax(logits_sm,dim=-1)
        else:
            logits_sampled = torch.multinomial(logits_sm,1)

        logits_byte_txt = logits_sampled.tolist()
        text_result = text_result + decode(logits_byte_txt)
        byte_text = encode(text_result[max(0,len(text_result)-window):])
    
    return text_result

def __apply_top_k(logits : torch.Tensor, k : int):
    if k is None or k <= 0 or k >= logits.numel():
        return logits
    
    topk_vals, _= torch.topk(logits, k)

    threshold = topk_vals[-1] 

    masked = logits.clone()
    masked[masked < threshold] = float("-inf")
    return masked

def __apply_top_p(logits : torch.Tensor, p : float): # Revisar
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


def create_run_features(model : torch.nn.Module,run_date : str ,lr : float, batch_size : int, wd : float, path: str):
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
        featuresfile.write("date: " +  run_date + "\n")
        featuresfile.write("lr: " +  str(lr) + "\n")
        featuresfile.write("batch_size: " +  str(batch_size) + "\n")
        featuresfile.write("weight_decay: " +  str(wd) + "\n")