from src.data import generate_data

def test():
    window = 5
    data, _ , _ = generate_data(window=window, eval_thr = 0)
    x,y = data[0]
    x2,y2 = data[(int)(len(data)/2)]
    x3,y3 = data[-1]
    assert x[1] == y[0] and x[3] == y[2] and x[-1] == y[-2] 
    assert x2[1] == y2[0] and x2[3] == y2[2] and x2[-1] == y2[-2] 
    assert x3[1] == y3[0] and x3[3] == y3[2] and x3[-1] == y3[-2] 


    