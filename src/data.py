from torch.utils.data import TensorDataset, DataLoader
import torch
from . import utils
def generate_data(window : int, stride : int = 1, file : str = "./tiny_shakespare.txt", eval_thr : float = 0.1) -> tuple[TensorDataset,TensorDataset]:
    """Generates a dataset with the *.txt file given

    Parameters
    ----------
    window : int
        The window which is used to slice the text.
    stride : int
        The stride between each window
    file : str
        The path of the text file.
    eval_thr : float
        The threshold of the eval part of the text.

    Returns
    -------
    A tuple composed of the train_dataset and eval_dataset, the size of de eval_dataset will be the eval_the porcent of the windows,
    and the size of the train_dataset will be the remaining windows.
    """
    if(eval_thr > 1 or eval_thr < 0):
        raise ValueError("Invalid threshold, the threshold must be a value between 0 and 1")
    
    print("file =", file, "type =", type(file), )
    f = open(file)
    txt = f.read()

    min_thr = min((1-eval_thr),eval_thr)

    if len(txt) < window or (min_thr > 0 and len(txt)*min_thr < window): 
        print(len(txt))
        raise ValueError("Window too big for this text and threshold ")

    data = utils.encode(txt)

    cut = (int)(len(data)*(1-eval_thr))

    data_train = torch.tensor(data[:cut]) # (a,b,c)
    data_eval = torch.tensor(data[cut:]) # (....,d,e,f)

    # Dataset creations

    train_dataset = None
    eval_dataset = None
    
    if (1-eval_thr > 0): # Train
        X_data_train  = data_train[:-1].unfold(-1,window,stride)
        Y_data_train  = data_train[1:].unfold(-1,window,stride)
        train_dataset = TensorDataset(X_data_train,Y_data_train)

    if (eval_thr > 0): # Eval
        X_data_eval  = data_eval[:-1].unfold(-1,window,stride)
        Y_data_eval  = data_eval[1:].unfold(-1,window,stride)
        eval_dataset = TensorDataset(X_data_eval,Y_data_eval)

    print(f"Train_data shape: {X_data_train.shape}")
    return train_dataset , eval_dataset














