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


def gen_text(model : torch.nn.Module , text_start : str , ntokens : int ,temperature : float, top_k : float, top_p : float, device : torch.device = get_device()):

    model.eval()

    text_result : str = text_start
    byte_text = encode(text_start)
    window = len(text_start)

    for _ in range(0,ntokens):
        t_text_start = torch.tensor([byte_text],device=device)
        # [126,255]
        pretemperature_logits = torch.squeeze(model(t_text_start),dim=0)
        #print(logits)
        logits = torch.div(pretemperature_logits,temperature)

        k_logits = __apply_top_k(logits,top_k)
        p_logits = __apply_top_p(k_logits, top_p, device)

        logits_sm = torch.softmax(p_logits,dim=-1)
        #print(logits_sm)
        logits_sampled = torch.multinomial(logits_sm,1)
        #print(logits_argmax)
        logits_bytes_txt = torch.squeeze(logits_sampled).tolist()
        #print(lista_txt)
        #print("TEXT:",decode(logits_bytes_txt))
        text_result = text_result + decode([logits_bytes_txt[-1]])
        #print("NEXTTEXT",len(text_result[len(text_result)-window:]))
        byte_text = encode(text_result[len(text_result)-window:])
    
    print(text_result)

def __apply_top_k(logits : torch.Tensor, k : int):
    if k is None or k <= 0 or k >= logits.numel():
        return logits
    
    topk_vals, _= torch.topk(logits, k)

    threshold = topk_vals[:,-1] 
    threshold = threshold.reshape(-1,1)
    threshold = threshold.expand((-1,logits.shape[-1]))

    masked = logits.clone()
    masked[masked < threshold] = float("-inf")
    return masked

def __apply_top_p(logits : torch.Tensor, p : float, device : torch.device = get_device()): # Revisar
    if p is None or p <= 0 or p >= 1:
        return logits

    probs = torch.softmax(logits, dim=-1) 

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    remove = cum_probs > p
    remove[:,0] = False

    masked = logits.clone()

    remove_vocab_mask = torch.zeros([masked.shape[0],masked.shape[1]], dtype=torch.bool, device=device)
    # Writes in the mask the values of the "remove" in the given index ubications
    # [2,0,3,1] and [false, false, true, true] => [true,false,true,false]
    remove_vocab_mask.scatter_(dim=1, index=sorted_idx, src=remove) 

    masked_logits = masked.masked_fill_(remove_vocab_mask,float('-inf'))
    return masked_logits