import torch,argparse
from torch.utils.data import DataLoader
from src import utils,models,train,data

#ARGS 
parser = argparse.ArgumentParser()
# Training
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--eval-metrics", action="store_true")
parser.add_argument("--device", type=str, default=utils.get_device())
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--weight-decay", type=float, default=0.1)
# Generation
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--preset", type=str,  default="default", choices=["default","short_stable","creative","debug_greedy"])
parser.add_argument("--temperature", type=float, default=None)
parser.add_argument("--top-p", type=float, default=None)
parser.add_argument("--top-k", type=int, default=None)

args = parser.parse_args()

#BASIC CONFIG
utils.set_seed(0,deterministic=True)

ckpt_path = args.ckpt_path

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

if (ckpt_path is None):
    model = models.mini_GPT() 
else:
    ckpt, model = utils.load_checkpoint(path=ckpt_path)
model = model.to(device)
opt = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=weight_decay)
if ckpt_path is not None and ckpt["optimizer"] is not None: opt.load_state_dict(ckpt["optimizer"])
loss_fn = torch.nn.CrossEntropyLoss().to(device)

## WARMUP AND SCHEDULER
if max_steps is not None and epochs is not None:
    planned_steps = min(max_steps,len(dataloader_train)*epochs)
elif max_steps is not None:
    planned_steps = max_steps
elif epochs is not None:
    planned_steps = len(dataloader_train) * epochs
else:
    raise ValueError("You must specify either --max-steps or --epochs")

    

warmup = True
warmup_steps = (int)(0.05*planned_steps) # The number of warmup_steps is the 5% of total steps
warmuper = train.warmup(opt,lr,warmup_steps) if warmup else None

scheduler = True
scheduler_steps = planned_steps-warmup_steps
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=scheduler_steps) if scheduler else None

#TRAIN OR EVAL
print(f"Used device: {device}")
if args.eval_only:
    if ckpt_path is None:
        raise ValueError("You should specify --ckpt-path when using --eval-only") 
    if args.eval_metrics:
        val_metrics = train.evaluate(model,dataloader_eval,loss_fn,device) 
        print("EVALUATION")
        print(f"Eval - Loss: {val_metrics['eval_loss']:.4f}, Acc: {val_metrics['eval_acc']:.4f}")
    if (args.eval_metris is None and len(prompt) == 0):
        print("Checkpoint loaded. No evaluation or generation requested.")
else:
    train.fit(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,max_steps=max_steps,early_stopping=patience,scheduler=scheduler,warmuper=warmuper)

print(len(prompt)) 
if(len(prompt)>0):   
    print(utils.gen_text(model,prompt,window,new_tokens,temperature,top_k,top_p,device=device,preset=preset))



