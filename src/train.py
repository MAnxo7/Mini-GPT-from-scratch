import torch
import os,csv,datetime, time
from . import utils,viz
#CSV: epoch, split, loss, acc, lr, time.
class warmup():
    def __init__(self,opt,lr_target,warmup_steps):
        self.lr_target = lr_target
        self.warmup_steps = warmup_steps
        self.opt = opt
        self.act_steps = 0
    
    def is_finished(self):
        return self.act_steps >= self.warmup_steps
    
    def step(self):
        if(self.is_finished()):
            raise RuntimeError("Step try after the warmup is finished")
        self.opt.param_groups[0]['lr'] = self.lr_target * ((self.act_steps+1) / self.warmup_steps)
        self.act_steps+=1
        
    

def fit(model, device, train_loader, val_loader, optimizer, loss_fn, epochs, scheduler = None, warmuper = None, early_stopping=None, run_dir=os.path.join(".","runs")):
    if epochs <= 0:
        raise ValueError("Epochs can't be 0 or negative. Try increasing --epochs or using --eval-only")
    print(device)
    act_epoch,last_improve = 0,0
    pre_eval_loss = None
    vpatience = early_stopping if early_stopping is not None else float("inf") 
    ## WARMUP CREATION


    run_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    thisrun_path = os.path.join(run_dir,run_date)
    os.makedirs(thisrun_path,exist_ok=True)
    os.makedirs(os.path.join(thisrun_path,"figures"),exist_ok=True)
    csv_path = os.path.join(thisrun_path,"metrics.csv")
    last_ckpt_path = os.path.join(thisrun_path,"last.pt")
    best_ckpt_path = os.path.join(thisrun_path,"best.pt")
    best_eval_loss, best_eval_acc, best_train_loss, best_train_acc = float("inf"), 0.0, float("inf"), 0.0
    epoch_time_list = []
    # CSV Head
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["epoch","split","loss","acc","lr","duration_s"])
    while(act_epoch < epochs and last_improve < vpatience):
        print("epoch nº",act_epoch)

        #print("lr",optimizer.param_groups[0]['lr'])
        #TRAIN
        t0 = time.time()
        train_metrics = train_one_epoch(model,train_loader,optimizer,loss_fn,device,scheduler=scheduler, warmuper=warmuper)
        train_time = time.time() - t0
        #EVALUATE 
        t0 = time.time()
        eval_metrics = evaluate(model,val_loader,loss_fn,device)
        eval_time = time.time() - t0
        #SAVE EPOCH TIME
        epoch_time_list.append(train_time+eval_time)
        #WARMUP AND SCHEDULER 

        # IS THE BEST?
        if (best_eval_loss > eval_metrics["eval_loss"]):
            best_eval_loss = eval_metrics["eval_loss"]
            best_train_loss = train_metrics["train_loss"]
            best_loss_epoch = act_epoch
        if (best_eval_acc < eval_metrics["eval_acc"]):
            best_eval_acc = eval_metrics["eval_acc"]
            best_train_acc = train_metrics["train_acc"]
            best_acc_epoch = act_epoch
        #SAVE DATA IN CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=",")
            writer.writerow([act_epoch,"train",train_metrics["train_loss"],train_metrics["train_acc"],optimizer.param_groups[0]["lr"],train_time])
            writer.writerow([act_epoch,"eval",eval_metrics["eval_loss"],eval_metrics["eval_acc"],optimizer.param_groups[0]["lr"],eval_time])
        #UPDATE LOOP
        if pre_eval_loss is not None and eval_metrics["eval_loss"] >= pre_eval_loss:
            last_improve+=1
        else:
            utils.save_checkpoint(model,optimizer,act_epoch,best_ckpt_path,scheduler)
            last_improve=0
        pre_eval_loss = eval_metrics["eval_loss"]
        utils.save_checkpoint(model,optimizer,act_epoch,last_ckpt_path,scheduler)
        act_epoch+=1
    if last_improve >= vpatience:
        utils.load_checkpoint(best_ckpt_path,model,optimizer)
        
    viz.plot_from_csv(csv_path)
    
    gap_best_loss = best_eval_loss-best_train_loss
    gap_best_acc = best_train_acc-best_eval_acc
    avg_epoch_time = sum(epoch_time_list) / len(epoch_time_list)
    print("-----------------")
    print(f"Best eval loss: {best_eval_loss:.4f} | Best train loss: {best_train_loss:.4f} | GAP: {gap_best_loss:.4f} | Epoch: {best_loss_epoch}")
    print(f"Best eval acc : {best_eval_acc:.4f} | Best train acc : {best_train_acc:.4f} | GAP: {gap_best_acc:.4f} | Epoch: {best_acc_epoch}")
    print(f"Average epoch time: {avg_epoch_time:.4f}")
                
        
def train_one_epoch(model, loader, optimizer,  loss_fn, device, scheduler = None, warmuper : warmup = None): 
    model.train()
    train_loss,train_acc,n_samples = 0.0,0.0,0
    step = 0
    for xn,yn in loader:
        xn, yn = xn.to(device), yn.to(device)  
        

        optimizer.zero_grad()
        logits = model(xn)
        # Reshaping 
        logits = torch.reshape(logits,(-1,logits.shape[-1]))
        yn = torch.reshape(yn,(-1,))

        loss = loss_fn(logits,yn)
        loss.backward()
        #for name,param in model.named_parameters():
        #   print(name,param.grad.norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient Clipping
        

        # OPTIMIZER, SCHEDULER AND WARMUP
        if warmuper is not None and not warmuper.is_finished():    
            warmuper.step()
            optimizer.step()
        elif scheduler is not None:
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                raise AttributeError("ReduceLROnPlateau scheduler isn't compatible with this model.")
            else:
                scheduler.step()
        else:
            optimizer.step()

        #Metrics
        samples = xn.size(0)
        act_loss = loss.item()
        train_loss += act_loss*samples
        act_acc = utils.accuracy_from_logits(logits, yn)
        train_acc += act_acc*samples
        n_samples+=samples
        step+=1
        if step%200 == 0: 
            print((step/len(loader))*100,"%")
            print("TRAIN_LOSS: ",act_loss," TRAIN ACC: ",act_acc," LR: ",optimizer.param_groups[0]['lr'])
    return {"train_loss":train_loss/n_samples,"train_acc":train_acc/n_samples}
        
        
def evaluate(model,loader , loss_fn, device):
    model.eval()
    eval_loss,eval_acc,n_samples = 0.0,0.0,0
    with torch.no_grad():
        for xn,yn in loader:
            xn, yn = xn.to(device), yn.to(device)  
            logits = model(xn)
            # Reshaping 
            logits = torch.reshape(logits,(-1,logits.shape[-1]))
            yn = torch.reshape(yn,(-1,))

            loss = loss_fn(logits,yn)
            #Metrics
            samples = xn.size(0)
            eval_loss += loss.item()*samples
            eval_acc += utils.accuracy_from_logits(logits, yn)*samples
            n_samples+=samples
    return {"eval_loss":eval_loss/n_samples,"eval_acc":eval_acc/n_samples}


# -------------------------- TRAIN WITH STEPS -----------------------------------------


def fit_steps(model, device, train_loader, val_loader, optimizer, loss_fn, epochs ,max_steps = None, scheduler = None, warmuper = None, early_stopping=None, run_dir=os.path.join(".","runs")):
    if epochs <= 0:
        raise ValueError("Epochs can't be 0 or negative. Try increasing --epoch or using --eval-only")
    if max_steps and max_steps <= 0:
        raise ValueError("Max_steps can't be 0 or negative. Try increasing --steps or using --eval-only")
    print(device)

    N_STEPS = 20
    STEP_MODE = True # This makes the x-axis of the accuracy and loss graphics created by matplot be in range of N_STEPS instead of range of epochs
    act_step,act_epoch,last_improve= 0,0,0
    train_time = 0
    pre_eval_loss = None
    vpatience = early_stopping if early_stopping is not None else float("inf") 

    run_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    thisrun_path = os.path.join(run_dir,run_date)
    os.makedirs(thisrun_path,exist_ok=True)
    os.makedirs(os.path.join(thisrun_path,"figures"),exist_ok=True)
    csv_path = os.path.join(thisrun_path,"metrics.csv")
    last_ckpt_path = os.path.join(thisrun_path,"last.pt")
    best_ckpt_path = os.path.join(thisrun_path,"best.pt")
    best_eval_loss, best_eval_acc, best_train_loss, best_train_acc = float("inf"), 0.0, float("inf"), 0.0
    epoch_time_list = []

    # CSV Head
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["step","split","loss","acc","lr","duration_s"])
    # TRAIN
    while(act_epoch < epochs and (not max_steps or act_step < max_steps) and last_improve < vpatience):

        print("#### epoch nº ", act_epoch, " ####")
        #TRAIN
        for xn,yn in train_loader:
            t0 = time.time()
            train_metrics = train_one_step(model,xn,yn,optimizer,loss_fn,device,scheduler=scheduler, warmuper=warmuper)
            train_time += time.time() - t0
            #EVALUATE 
            if act_step == 0 or act_step%N_STEPS == 0 or act_step >= max_steps:
                print("-- step nº",act_step," --")
                t0 = time.time()
                eval_metrics = evaluate(model,val_loader,loss_fn,device)
                eval_time = time.time() - t0
                #SAVE EPOCH TIME
                epoch_time_list.append(train_time+eval_time)
                train_time = 0
                # INTER-EPOCH STATS
                print("TRAIN_LOSS: ",train_metrics["train_loss"]," TRAIN_ACC: ",train_metrics["train_acc"],
                "\nEVAL_LOSS: ",train_metrics["train_loss"]," EVAL_ACC: ",eval_metrics["eval_acc"],
                "\nLR: ",optimizer.param_groups[0]['lr'])
                # IS THE BEST?
                if (best_eval_loss > eval_metrics["eval_loss"]):
                    best_eval_loss = eval_metrics["eval_loss"]
                    best_train_loss = train_metrics["train_loss"]
                    best_loss_epoch = act_step
                if (best_eval_acc < eval_metrics["eval_acc"]):
                    best_eval_acc = eval_metrics["eval_acc"]
                    best_train_acc = train_metrics["train_acc"]
                    best_acc_epoch = act_step
                #SAVE DATA IN CSV
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile,delimiter=",")
                    writer.writerow([act_step,"train",train_metrics["train_loss"],train_metrics["train_acc"],optimizer.param_groups[0]["lr"],train_time])
                    writer.writerow([act_step,"eval",eval_metrics["eval_loss"],eval_metrics["eval_acc"],optimizer.param_groups[0]["lr"],eval_time])
                #UPDATE LOOP
                if pre_eval_loss is not None and eval_metrics["eval_loss"] >= pre_eval_loss:
                    last_improve+=1
                else:
                    utils.save_checkpoint(model,optimizer,act_step,best_ckpt_path,steps_mode=STEP_MODE,extra=scheduler)
                    last_improve=0
                pre_eval_loss = eval_metrics["eval_loss"] 
                utils.save_checkpoint(model,optimizer,act_step,last_ckpt_path,steps_mode=STEP_MODE,extra=scheduler)
                if(act_step >= max_steps):
                    break
            act_step+=1
        act_epoch+=1           
    if last_improve >= vpatience:
        utils.load_checkpoint(best_ckpt_path,model,optimizer)
        
    viz.plot_from_csv(csv_path,step_mode=STEP_MODE)
    
    gap_best_loss = best_eval_loss-best_train_loss
    gap_best_acc = best_train_acc-best_eval_acc
    avg_cycle_time = sum(epoch_time_list) / len(epoch_time_list)
    print("-----------------")
    print(f"Best eval loss: {best_eval_loss:.4f} | Best train loss: {best_train_loss:.4f} | GAP: {gap_best_loss:.4f} | Epoch: {best_loss_epoch}")
    print(f"Best eval acc : {best_eval_acc:.4f} | Best train acc : {best_train_acc:.4f} | GAP: {gap_best_acc:.4f} | Epoch: {best_acc_epoch}")
    print(f"Average epoch time: {avg_cycle_time:.4f}")



def train_one_step(model, xn, yn, optimizer,  loss_fn, device, scheduler = None, warmuper : warmup = None): 
    model.train()
    xn, yn = xn.to(device), yn.to(device)  
    optimizer.zero_grad()
    logits = model(xn)
    # Reshaping 
    logits = torch.reshape(logits,(-1,logits.shape[-1]))
    yn = torch.reshape(yn,(-1,))

    loss = loss_fn(logits,yn)
    loss.backward()
    #for name,param in model.named_parameters():
    #   print(name,param.grad.norm())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient Clipping
        

    # OPTIMIZER, SCHEDULER AND WARMUP
    if warmuper is not None and not warmuper.is_finished():    
        warmuper.step()
        optimizer.step()
    elif scheduler is not None:
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise AttributeError("ReduceLROnPlateau scheduler isn't compatible with this model.")
        else:
            scheduler.step()
    else:
        optimizer.step()

    #Metrics

    act_loss = loss.item()

    act_acc = utils.accuracy_from_logits(logits, yn)

    return {"train_loss":act_loss,"train_acc":act_acc}