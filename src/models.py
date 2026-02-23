import torch

class BasicNN(torch.nn.Module):
    def __init__(self,neurons,in_features,out_features,device):
        super().__init__()
        H = neurons
        input = in_features
        output = out_features
        self.capa1 = torch.nn.Linear(input,H).to(device)
        self.capa2 = torch.nn.Linear(H,output).to(device)
        self.capas = [self.capa1,self.capa2]
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
                   
    def forward(self,x): 
        input = self.activation(self.capas[0](x))
        # Calculate and activate all the layers until the n-1 layer
        for i in range(1,len(self.capas)-1):
            input = self.activation(self.capas[i](input))
        # Returns the last layer without activation
        return self.capas[len(self.capas)-1](input)

class mini_GPT(torch.nn.Module):
    def __init__(self, X_data, device):
        super().__init__()
        VOCAB = 256
        d_model = 256
        context_L = 256
        n_heads = 8
        n_layers = 6
        d_ff = 4*d_model  #(MLP interno)
        dropout = 0.1

        self.embeds = torch.nn.Embedding(VOCAB,d_model,padding_idx=0,dtype = torch.long).to(device)
        self.pos_embeds = torch.nn.Embedding(context_L,d_model,dtype = torch.long)

        self.encoder = torch.nn.Enco
    
    def __encode__(text:str) -> list:
       text = text.encode(encoding="utf-8",errors="replace")
       return list(text)

    def __decode__(list_bytes:list) -> str:
        return str(list_bytes)