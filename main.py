import torch,argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src import utils,models,train,data

#ARGS 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--device", type=str, default=utils.get_device())
parser.add_argument("--ckpt-path", type=str, default=None)
args = parser.parse_args()

#BASIC CONFIG
utils.set_seed(0,deterministic=True)


epochs = args.epochs
batch = args.batch_size
lr = args.lr
device = args.device

# WARMUP SET (This avoid division by 0 if the epochs are too low)
if(epochs * 0.05 >= 1):
    warmup = True
else:
    warmup = False

# DATALOADERS CREATION
window = 256
numworkers = 10
persistent_workers = True
pin_memory = True
file = "./tiny_shakespare_little.txt"

dataset_train, dataset_eval = data.generate_data(window,file=file)
print(dataset_train)
dataloader_train = DataLoader(dataset=dataset_train,batch_size = batch,num_workers=10,persistent_workers=True,pin_memory=True)
dataloader_eval = DataLoader(dataset=dataset_eval,batch_size = batch,num_workers=10,persistent_workers=True,pin_memory=True)


model = models.mini_GPT(device,dropout=0.1)

weight_decay = 0
opt = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*0.95+1) #0.95 for the warmup
loss_fn = torch.nn.CrossEntropyLoss().to(device)

 #TRAIN OR EVAL
if args.eval_only:
    if args.ckpt_path is None:
        raise ValueError("You should specific --ckpt-path when use --eval-only")
    utils.load_checkpoint(args.ckpt_path,model,opt)
    val_metrics = train.evaluate(model,dataloader_eval,loss_fn,device) # MODIFICADO, DEBERIA SER DATALOADER_EVAL
    print(f"Eval - Loss: {val_metrics['eval_loss']:.4f}, Acc: {val_metrics['eval_acc']:.4f}")
    
else:
    train.fit(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,early_stopping=100,scheduler=None,warmup=False)

preds = "First Citizen:\n\
We are accounted poor citizens, the patricians good.\n\
What authority surfeits on would relieve us: if they\n\
would yield us but the superfluity, while it were\n\
wholesome, we might guess they relieved us humanely;\n\
but they think we are too dear: "
     
utils.gen_text(model,preds,200,device=device)
     # We are accounted poor citizens, the patricians good.
     # What authority surfeits on would relieve us: 



