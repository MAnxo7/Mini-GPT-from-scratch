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


def gen_text(model : torch.nn.Module , text_start : str , nletters : int , device : torch.device = get_device()):

    model.eval()

    text_result : str = text_start
    byte_text = encode(text_start)
    window = len(text_start)

    for _ in range(0,nletters):
        t_text_start = torch.tensor([byte_text],device=device)
        logits = torch.squeeze(model(t_text_start),dim=0)
        #print(logits)
        logits_sm = torch.softmax(logits,dim=-1)
        #print(logits_sm)
        logits_argmax = torch.argmax(logits_sm,dim=1)
        #print(logits_argmax)
        logits_bytes_txt = torch.squeeze(logits_argmax).tolist()
        #print(lista_txt)
        #print("TEXT:",decode(logits_bytes_txt))
        text_result = text_result + decode([logits_bytes_txt[-1]])
        #print("NEXTTEXT",len(text_result[len(text_result)-window:]))
        byte_text = encode(text_result[len(text_result)-window:])
    
    print(text_result)

