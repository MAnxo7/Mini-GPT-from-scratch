import torch
from src import utils,models,train
from src.data import generate_data
    
def test():
    # The model can memorize the predictions of a 16-length text. 
    utils.set_seed(0,deterministic=True)

    epochs = 15
    lr = 3e-4
    device = torch.device("cpu")
    window = 16

    dataset_train, _ , _= generate_data(window,eval_thr=0)

    xn , yn = dataset_train[0]

    model = models.mini_GPT(dropout=0).to(device)

    opt = torch.optim.Adam(params=model.parameters(),lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    for _ in range(0,epochs):
        train.train_one_step(model,xn,yn,opt,loss_fn,device)

    tensor_test = torch.unsqueeze(xn,dim=0)
    logits = model(tensor_test)

    logits = torch.reshape(logits,(-1,logits.shape[-1]))
    logits_sm = torch.softmax(logits,dim=-1)
    preds = torch.argmax(logits_sm,dim=-1)
    
    print("XN: ",utils.byte_decode(xn),"PREDS: ",utils.byte_decode(preds)) 

    data = train.train_one_step(model, xn, yn, opt, loss_fn, device)
    print(data)
    assert data["train_loss"] < 0.1 and data["train_acc"] > 0.95