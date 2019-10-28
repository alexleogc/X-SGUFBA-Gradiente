import numpy as np

def predict (beta,X):
    """
    Calculo matricial de uma função linear:
    
    X - matriz das observações; e
    beta - Variáveis da função calculada
    """
    
    X = np.asarray(X) ; beta = np.asarray(beta)
    return X.dot(beta)

def cost(theta,X,y):
    """
    Função custo que computa os desvios entre o dado observado e o calculado:
    X - Matriz das variáveis observadas (nxm : n - número de variáveis ; m - número de observações)
    y - Vetor de com as respostas esperadas (tamanho m) 
    """
    return np.sum((y-predict(theta,X))**2)/len(y)


def gradient_descent(X,y,beta,alpha=0.01,ninter=100):
    
    """
    Ex:
    
    beta_calc, cost_hist, beta_hist = gradient_descent(X,y,beta_ini,alpha=0.1,ninter=10)
    
    """
    
    try: X = np.append(X.reshape((len(X),1)),(np.ones((len(X),1))),axis=1)
    except: X = np.append(X,(np.ones((len(X),1))),axis=1)

    beta = np.asarray(beta)
    cost_history = np.zeros(ninter) ; beta_history = np.zeros((ninter,len(beta)))
    
    for i in range(ninter):
        beta = beta- alpha*(X.T.dot((predict(beta,X))-y))*(2/len(y))
        beta_history[i,:] = beta
        cost_history[i] = cost(beta,X,y)
        
    return beta, cost_history, beta_history