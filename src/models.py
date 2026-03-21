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
    def __init__(self, device, dropout = 0.1):
        super().__init__()
        VOCAB = 256
        d_model = 256
        context_L = 256
        self.n_heads = 8
        n_layers = 6
        d_ff = 4*d_model  #(MLP interno)
        dropout = dropout

        self.embeds_layer = torch.nn.Embedding(VOCAB,d_model,dtype = torch.float).to(device)
        self.pos_embeds_layer = torch.nn.Embedding(context_L,d_model,dtype = torch.float).to(device)
        
        # I use encoder in Pytorch because Pytorch doen't have a decoder_only layer. The torch.nn.TransformerDecoderLayer is equivalent to Encoder-Decoder instead
        # decoder only.
        decoder_layer = torch.nn.TransformerEncoderLayer(d_model,self.n_heads,d_ff,dropout,batch_first=True).to(device) 
        self.decoder = torch.nn.TransformerEncoder(decoder_layer,n_layers).to(device)

        for p in self.decoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.linear_layer = torch.nn.Linear(in_features=d_model,out_features=VOCAB).to(device)
    
    def forward(self, X : torch.Tensor):

        positions = torch.arange(0,X.shape[-1]).to(X.device)

        X = self.embeds_layer(X)
        P = self.pos_embeds_layer(positions)

        X = torch.add(X,P)

        causal_mask = positions.view(1,-1) > positions.view(-1,1)
        X = self.decoder(X,mask=causal_mask) 
        
        X = self.linear_layer(X)
        return X



    
