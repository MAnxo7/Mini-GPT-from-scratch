from torch.utils.data import TensorDataset, DataLoader
import torch
from . import utils
def generate_data(window : int, file : str = "./tiny_shakespare.txt", eval_thr : float = 0.1) -> tuple[TensorDataset,TensorDataset]:
    """Generates a dataset with the *.txt file given

    Parameters
    ----------
    window : int
        The window which is used to slice the text.
    file : str
        The path of the text file.
    eval_thr : float
        The threshold of the eval part of the text.

    Returns
    -------
    A tuple composed of the train_dataset and eval_dataset, the size of de eval_dataset will be the eval_the porcent of the text,
    and the size of the train_dataset will be the remaining text.
    """
    if(eval_thr >= 1):
        raise ValueError("Invalid threshold, the threshold can't be greater or equal to 1.")
    print("file =", file, "type =", type(file), )
    f = open(file)
    txt = f.read()

    print(len(txt)*eval_thr)
    if (len(txt) < window or (eval_thr > 0 and (len(txt)*eval_thr < window or len(txt)*(1-eval_thr) < window))): #Arreglar, no va bien con lineas
        raise ValueError("Window too big for this text and threshold ")

    # EVAL DATASET
    if (eval_thr > 0):
        txt_lines = txt.splitlines()
        list_empty_lines = __empty_lines(txt_lines)
        thr_index = (int)((1-eval_thr)*len(list_empty_lines))
        line_of_cut = list_empty_lines[thr_index]
        delimiter = "\n"
        txt_eval = delimiter.join(txt_lines[line_of_cut:])
        data_eval = utils.encode(txt_eval)

        #X_data_eval  = torch.tensor(data_eval[:-1],dtype=torch.long)
        #Y_data_eval  = torch.tensor(data_eval[1:],dtype=torch.long)
        #X_data_eval  = X_data_eval.unfold(0,window,1) # (a,b,c,d,e),(b,c,d,e,f)
        #Y_data_eval  = Y_data_eval.unfold(0,window,1) # (b,c,d,e,f),(c,d,e,f,g)

        eval_dataset = TensorDataset(X_data_eval,Y_data_eval)
        txt_train = delimiter.join(txt_lines[:line_of_cut])
    else:
        eval_dataset = None
        txt_train = txt
    # TRAIN DATASET
    
    data_train = utils.encode(txt_train)
    X_data_train  = torch.tensor(data_train[:-1],dtype=torch.long)
    Y_data_train  = torch.tensor(data_train[1:],dtype=torch.long)

    X_data_train  = X_data_train.unfold(0,window,window-1) # (a,b,c,d,e),(e,f,g,h,i)
    Y_data_train  = Y_data_train.unfold(0,window,window-1) # (b,c,d,e,f),(f,g,h,i,j)
    train_dataset = TensorDataset(X_data_train,Y_data_train)

    print(X_data_train.shape)
    return train_dataset , eval_dataset


def __empty_lines(lines):
    return [i for i, line in enumerate(lines) if line.strip() == ""]












