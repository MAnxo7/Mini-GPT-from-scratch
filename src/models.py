import torch

class mini_GPT(torch.nn.Module):
    def __init__(self, device, dropout = 0.1):
        super().__init__()
        
        self.VOCAB = 256
        self.d_model = 256
        self.context_L = 256
        self.n_heads = 8
        self.n_layers = 6
        self.d_ff = 4*self.d_model  #(MLP interno)
        self.dropout = dropout

        self.embeds_layer = torch.nn.Embedding(self.VOCAB,self.d_model,dtype = torch.float).to(device)
        self.pos_embeds_layer = torch.nn.Embedding(self.context_L,self.d_model,dtype = torch.float).to(device)
        
        # I use encoder in Pytorch because Pytorch doen't have a decoder_only layer. The torch.nn.TransformerDecoderLayer is equivalent to Encoder-Decoder instead
        # decoder only.
        decoder_layer = torch.nn.TransformerEncoderLayer(self.d_model,self.n_heads,self.d_ff,dropout,batch_first=True).to(device) 
        self.decoder = torch.nn.TransformerEncoder(decoder_layer,self.n_layers).to(device)

        for p in self.decoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.linear_layer = torch.nn.Linear(in_features=self.d_model,out_features=self.VOCAB).to(device)
    
    def forward(self, X : torch.Tensor):

        positions = torch.arange(0,X.shape[-1]).to(X.device)

        X = self.embeds_layer(X)
        P = self.pos_embeds_layer(positions)

        X = torch.add(X,P)

        causal_mask = positions.view(1,-1) > positions.view(-1,1)
        X = self.decoder(X,mask=causal_mask) 
        
        X = self.linear_layer(X)
        return X



    
