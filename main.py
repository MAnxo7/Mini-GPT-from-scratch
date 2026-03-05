import torch,argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src import utils,models,train,data

#ARGS 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
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


# DATALOADERS CREATION
window = 5
numworkers = 10
persistent_workers = True
pin_memory = True

dataset_train, _ = data.generate_data(window,device)
dataset_eval, _ = data.generate_data(window,device)

dataloader_train = DataLoader(dataset=dataset_train,batch_size = batch,num_workers=10,persistent_workers=True,pin_memory=True)
dataloader_eval = DataLoader(dataset=dataset_eval,batch_size = batch,num_workers=10,persistent_workers=True,pin_memory=True)

model = models.mini_GPT(device)

opt = torch.optim.Adam(params=model.parameters(),lr=lr)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

 #TRAIN OR EVAL
if args.eval_only:
    if args.ckpt_path is None:
        raise ValueError("You should specific --ckpt-path when use --eval-only")
    utils.load_checkpoint(args.ckpt_path,model,opt)
    val_metrics = train.evaluate(model,dataloader_eval,loss_fn,device)
    print(f"Eval - Loss: {val_metrics['eval_loss']:.4f}, Acc: {val_metrics['eval_acc']:.4f}")
    
else:
    train.fit(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,early_stopping=500)
    print(model("That sort was well ")) # That sort was well fished for.


