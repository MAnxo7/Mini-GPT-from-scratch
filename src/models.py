import torch

class mini_GPT(torch.nn.Module):
    def __init__(self, vocab_size : int = 257,dropout = 0.1, d_model = 256, n_layers = 6, n_heads = 8, context_L = 256, d_ff = None):
        """
        Create a small decoder-only GPT-style Transformer model for byte-level
        language modeling.

        The model receives token IDs in the range [0, 255], applies token and
        positional embeddings, processes the sequence with a causal Transformer
        stack, and projects the final hidden states to logits over the byte-level
        vocabulary.

        Parameters
        ----------
        vocab_size : int, default=257
            The number of diferent tokens the model can predict. 
            If it's a byte-level model a vocab_size of 257 will be used (256 of possible byte predictions and a reserved one for a possible padding update)
        dropout : float, default=0.1
            Dropout probability used inside the Transformer layers.

        d_model : int, default=256
            Hidden dimension of token embeddings and Transformer representations.

        n_layers : int, default=6
            Number of Transformer encoder layers used as decoder-only causal blocks.

        n_heads : int, default=8
            Number of attention heads in each Transformer layer.

        context_L : int, default=256
            Maximum context length supported by the positional embedding layer.

        d_ff : int or None, default=None
            Inner dimension of the feed-forward network inside each Transformer
            layer. If None, it is set to 4 * d_model.

        Notes
        -----
        The causal mask prevents each position from attending to future tokens.
        """
        super().__init__()
        
        self.VOCAB = vocab_size

        self.d_model = d_model
        self.context_L = context_L
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = 4*self.d_model if d_ff is None else d_ff  #(inner MLP)
        self.dropout = dropout

        self.model_config = {
            "vocab_size": self.VOCAB,
            "d_model": self.d_model,
            "context_L": self.context_L,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        }

        self.embeds_layer = torch.nn.Embedding(self.VOCAB,self.d_model,dtype = torch.float)
        self.pos_embeds_layer = torch.nn.Embedding(self.context_L,self.d_model,dtype = torch.float)
        
        # I use encoder in Pytorch because Pytorch doen't have a decoder_only layer. The torch.nn.TransformerDecoderLayer is equivalent to Encoder-Decoder instead
        # decoder only.
        decoder_layer = torch.nn.TransformerEncoderLayer(self.d_model,self.n_heads,self.d_ff,dropout,batch_first=True)
        self.decoder = torch.nn.TransformerEncoder(decoder_layer,self.n_layers)

        for p in self.decoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.linear_layer = torch.nn.Linear(in_features=self.d_model,out_features=self.VOCAB)
    


    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass of the model.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of byte token IDs with shape [batch_size, sequence_length].
            Each value must be in the range [0, vocab_size-1].

        Returns
        -------
        torch.Tensor
            Logits over the byte-level vocabulary with shape
            [batch_size, sequence_length, vocab_size].

            logits[:, t, :] represents the model's prediction for the next byte
            after position t. 
        """
        positions = torch.arange(0,X.shape[-1]).to(X.device)

        X = self.embeds_layer(X)
        P = self.pos_embeds_layer(positions)

        X = torch.add(X,P)

        causal_mask = positions.view(1,-1) > positions.view(-1,1)
        X = self.decoder(X,mask=causal_mask) 
        
        X = self.linear_layer(X)
        return X

    def get_config(self):
        return self.model_config

    
