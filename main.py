import torch,argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src import utils,models,train,data

#ARGS 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--max-steps", type=int, default=None)
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
max_steps = args.max_steps
lr = args.lr
device = args.device

patience = 200
# DATALOADERS CREATION
window = 128
stride = 8
numworkers = 10
persistent_workers = True
pin_memory = True
file = "./tiny_shakespare.txt"

dataset_train, dataset_eval = data.generate_data(window,file=file,stride=stride)
dataloader_train = DataLoader(dataset=dataset_train,batch_size = batch,num_workers=10,persistent_workers=True,pin_memory=True)
dataloader_eval = DataLoader(dataset=dataset_eval,batch_size = batch,num_workers=10,persistent_workers=True,pin_memory=True)


model = models.mini_GPT(device,dropout=0.1)

weight_decay = 0.05
opt = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

## WARMUP AND SCHEDULER
#total_steps = (int)((len(dataset_train)/batch)*epochs)
total_steps = epochs

warmup = True
warmup_steps = (int)(0.05*total_steps) # The number of warmup_steps is the 5% of total steps
warmuper = train.warmup(opt,lr,warmup_steps) if warmup else None

scheduler = True
scheduler_steps = total_steps-warmup_steps+1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=scheduler_steps) if scheduler else None


#TRAIN OR EVAL
if args.eval_only:
    if args.ckpt_path is None:
        raise ValueError("You should specific --ckpt-path when use --eval-only")
    utils.load_checkpoint(args.ckpt_path,model,opt)
    val_metrics = train.evaluate(model,dataloader_eval,loss_fn,device) # MODIFICADO, DEBERIA SER DATALOADER_EVAL
    print(f"Eval - Loss: {val_metrics['eval_loss']:.4f}, Acc: {val_metrics['eval_acc']:.4f}")
    
else:
    train.fit_steps(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,max_steps=max_steps,early_stopping=patience,scheduler=scheduler,warmuper=warmuper)

preds = "All things in common nature should produce\n\
Without sweat or endeavour: treason, felony,\n\
Sword, pike, knife, gun, or need of any "
print(len(preds))    
utils.gen_text(model,preds,200,device=device)
     # We are accounted poor citizens, the patricians good.
     # What authority surfeits on would relieve us: 



