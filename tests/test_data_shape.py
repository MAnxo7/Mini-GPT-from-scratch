from src.data import generate_data

def test():
    data, _ = generate_data(batch_size=16,window=5)
    X, _ = data[0]

    
