def set_seed(seed: int, deterministic: bool = False):
    import os, random, numpy as np, torch
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
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def accuracy_from_logits(logits, y_true):
    import torch
    logits_sm = torch.softmax(logits,dim=-1)
    preds = torch.argmax(logits_sm,dim=-1)
    #print("preds=",preds)
    #print("y_true=",y_true)
    correct = torch.sum((preds==y_true).int()).item()
    nelems = torch.numel(preds)
    return correct/nelems

def save_checkpoint(model, optimizer, epoch, path, extra: dict | None = None):
    import torch, os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "extra": extra or {},
    }
    torch.save(payload, path)
    
def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    import torch
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