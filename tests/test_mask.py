import torch
from src import utils,models,train,data

def test():
    # Token logits doesn't depend on consequent tokens. (The causal mask is being applied correctly)
    device = utils.get_device()

    model = models.mini_GPT(dropout=0).to(device)

    # The logits for "a" must be the same despite having a different sufix.
    texto1 = "a" + "bbbbbbbbbbbbbb"
    texto2 = "a" + "zzzzzzzzzzzzzz"
    
    if(len(texto1) != len(texto2)):
        raise ValueError("Different text lenghts on test")
    
    model.eval()

    tensor_test = torch.tensor(utils.byte_encode(texto1)).to(device)
    tensor_test = torch.unsqueeze(tensor_test,dim=0)
    logits1 = model(tensor_test)

    tensor_test = torch.tensor(utils.byte_encode(texto2)).to(device)
    tensor_test = torch.unsqueeze(tensor_test,dim=0)
    logits2 = model(tensor_test)

    thr = 1e-6
    assert torch.all(abs(logits1[0,0,:] - logits2[0,0,:]) < thr)