import torch,argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src import utils,models,train,data

#ARGS 
parser = argparse.ArgumentParser()
# Training
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--device", type=str, default=utils.get_device())
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--weight-decay", type=float, default=0.1)
# Generation
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--preset", type=str,  default="default", choices=["default","short_stable","creative","debug_greedy"])
parser.add_argument("--temperature", type=float, default=None)
parser.add_argument("--top-p", type=float, default=None)
parser.add_argument("--top-k", type=float, default=None)

args = parser.parse_args()

#BASIC CONFIG
utils.set_seed(0,deterministic=True)


epochs = args.epochs
batch = args.batch_size
max_steps = args.max_steps
lr = args.lr
weight_decay = args.weight_decay
device = args.device

prompt = args.prompt
new_tokens = args.max_new_tokens
preset = args.preset
temperature = args.temperature
top_p = args.top_p
top_k = args.top_k

patience = 300000

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
opt = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

## WARMUP AND SCHEDULER
planned_steps = min(max_steps if max_steps else float("inf"),len(dataloader_train)*epochs)

warmup = True
warmup_steps = (int)(0.05*planned_steps) # The number of warmup_steps is the 5% of total steps
warmuper = train.warmup(opt,lr,warmup_steps) if warmup else None

scheduler = True
scheduler_steps = planned_steps-warmup_steps
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=scheduler_steps) if scheduler else None


#TRAIN OR EVAL
print(device)
if args.eval_only:
    if args.ckpt_path is None:
        raise ValueError("You should specific --ckpt-path when use --eval-only") 
    utils.load_checkpoint(args.ckpt_path,model,opt)
    #val_metrics = train.evaluate(model,dataloader_eval,loss_fn,device) 
    #print("EVALUATION")
    #print(f"Eval - Loss: {val_metrics['eval_loss']:.4f}, Acc: {val_metrics['eval_acc']:.4f}")
    
else:
    train.fit_steps(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,max_steps=max_steps,early_stopping=patience,scheduler=scheduler,warmuper=warmuper)

print(len(prompt)) 
if(len(prompt)>0):   
    print(utils.gen_text(model,prompt,window,new_tokens,temperature,top_k,top_p,device=device,preset=preset))



