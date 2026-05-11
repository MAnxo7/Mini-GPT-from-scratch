

#BASIC CONFIG
def test():
    import torch
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    from src import utils,models,train,data
    
    utils.set_seed(0,deterministic=True)

    epochs = 15
    lr = 3e-4
    device = torch.device("cpu")

    # DATALOADERS CREATION
    window = 16

    dataset_train, _ = data.generate_data(window,file="./tiny_shakespare_little.txt",eval_thr=0)

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
    
    print("XN: ",utils.decode(xn),"PREDS: ",utils.decode(preds)) 

    data = train.train_one_step(model, xn, yn, opt, loss_fn, device)
    print(data)
    assert data["train_loss"] < 0.1 and data["train_acc"] > 0.95