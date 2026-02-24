from torch.utils.data import TensorDataset, DataLoader
import torch
from . import utils
def generate_data(window : int, batch_size : int, num_workers : int, persitent_workers : bool, pin_memory : bool) -> TensorDataset:

    f = open("/home/manxo/Escritorio/Mini-GPT/tiny_shakespare.txt")
    txt = f.read()
    data = utils.encode(txt)
    X_data = torch.tensor(data,dtype=torch.short)
    Y_data = torch.tensor(data[1:],dtype=torch.short)
    X_data = X_data.unfold(0,window,window-1) # (a,b,c,d,e),(e,f,g,h,i)
    Y_data = Y_data.unfold(0,window,window-1) # (b,c,d,e,f),(f,g,h,i,j)
    
    print(X_data[1],Y_data[1])
    # Datasets 
    X_dataset = TensorDataset(X_data)
    Y_dataset = TensorDataset(Y_data)

    # Dataloaders

    X_dataloader = DataLoader(X_dataset, batch_size, num_workers=num_workers, persistent_workers=persitent_workers, pin_memory=pin_memory)
    Y_dataloader = DataLoader(X_dataset, batch_size, num_workers=num_workers, persistent_workers=persitent_workers, pin_memory=pin_memory)

    return X_dataloader,Y_dataloader















