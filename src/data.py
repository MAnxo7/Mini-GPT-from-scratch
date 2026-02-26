from torch.utils.data import TensorDataset, DataLoader
import torch
from . import utils
def generate_data(window : int, file : str = "tiny_shakespare.txt") -> tuple[TensorDataset,str]:
    """Generates a dataset with the *.txt file given

    Parameters
    ----------
    window : int
        The window which is used to slice the text.
    batch_size : float
        The size of the batch.

    Returns
    -------
    A tuple composed of the dataset token-bytes and the text of the file given as string.
    """

    f = open(file)
    txt = f.read()
    data = utils.encode(txt)
    X_data = torch.tensor(data,dtype=torch.short)
    Y_data = torch.tensor(data[1:],dtype=torch.short)
    X_data = X_data.unfold(0,window,window-1) # (a,b,c,d,e),(e,f,g,h,i)
    Y_data = Y_data.unfold(0,window,window-1) # (b,c,d,e,f),(f,g,h,i,j)
    
    print(X_data.shape,Y_data.shape)
    # Datasets 
    dataset = TensorDataset(X_data,Y_data)

    return dataset , txt















