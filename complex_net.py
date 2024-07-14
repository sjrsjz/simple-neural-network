import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List,Callable

class ComplexLayer:
    def __init__(self, input_dim:int, output_dim:int,f:Callable[[List[float]],float],d_f:List[Callable[[List[float]],float]]):
        # 注意这里的f和d_f都是多元函数
        self.weights_count = len(d_f) # 权重矩阵的个数
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights:List[np.ndarray] = [np.random.rand(output_dim,input_dim)*2-1 for _ in range(self.weights_count)]
        self.f = f
        self.d_f = d_f

    def compute(self, inputs:List[float]) -> List[float]:
        dots = []
        for j in range(self.weights_count):
            dots.append(np.dot(self.weights[j],inputs))
        outputs = []
        for j in range(self.output_dim):
            outputs.append(self.f([dots[k][j] for k in range(self.weights_count)]))
        return outputs

    # 对于多元函数，可以使用以下的代码
    # inputs的类型为np.ndarray是一个一维数组
    def diff(self, inputs:np.ndarray, dim:int)->List[np.ndarray]:
        if dim < 0 or dim >= self.weights_count:
            raise Exception("dim out of range")
        dots = []
        # 这里计算第dim个权重矩阵的导数
        for j in range(self.weights_count):
            dots.append(np.dot(self.weights[j],inputs).tolist())

        # 计算第dim个权重矩阵的导数
        outputs = []
        for k in range(self.output_dim):
            dot = [dots[i][k] for i in range(self.weights_count)]
            d_f = self.d_f[dim](dot)
            tmp_matrix = np.zeros((self.output_dim,self.input_dim))
            for a in range(self.input_dim):
                for b in range(self.output_dim):
                    tmp_matrix[b][a] = d_f * (inputs[a] if b == k else 0)
            outputs.append(tmp_matrix)
        return outputs
    def diff_chain(self, inputs:np.ndarray, input_diffs:List[np.ndarray])->List[np.ndarray]:
        dots = []
        for j in range(self.weights_count):
            dots.append(np.dot(self.weights[j],inputs).tolist())
        
        outputs = []

        for k in range(self.output_dim):
            tmp_matrix = np.zeros((len(input_diffs[0]),len(input_diffs[0][0])))
            for n in range(self.weights_count):
                dot = [dots[i][k] for i in range(self.weights_count)]
                d_f = self.d_f[n](dot)
                for a in range(len(input_diffs[0][0])):
                    for b in range(len(input_diffs[0])):
                        for i in range(self.input_dim):
                            tmp_matrix[b][a] += d_f * self.weights[n][k][i] * input_diffs[i][b][a]
            outputs.append(tmp_matrix)
        return outputs
    
    def update_weights(self, diffs:List[np.ndarray], learning_rate:float):
        for j in range(self.weights_count):
            self.weights[j] -= learning_rate * diffs[j]

class ComplexNet:
    def __init__(self, layers:List[ComplexLayer], loss:Callable[[List[float],List[float]],float], d_loss:Callable[[List[float],List[float]],List[float]]):
        self.layers:List[ComplexLayer] = []
        for layer in layers:
            # copy
            tmp = ComplexLayer(layer.input_dim,layer.output_dim,layer.f,layer.d_f)
            tmp.weights = []
            for i in range(layer.weights_count):
                tmp.weights.append(layer.weights[i].copy())
            self.layers.append(tmp)
        self.loss = loss
        self.d_loss = d_loss
    def compute(self, inputs:List[float]) -> List[float]:
        tmp = inputs.copy()
        for layer in self.layers:
            tmp = layer.compute(tmp)
        return tmp
    def gradient_at_layer_k_of_dim_n(self, inputs:np.ndarray, dim:int, layer:int)->List[np.ndarray]:
        # 计算第layer层的第dim个权重矩阵的导数
        # inputs的类型为np.ndarray是一个一维数组
        # 使用链式法则计算，diff是最后一层的导数，diff_chain是前面的导数
        assert layer < len(self.layers)
        tmp = inputs.copy()
        # 计算第layer层的对权重的偏导
        for i in range(layer):
            tmp = self.layers[i].compute(tmp)
        curr = self.layers[layer].diff(tmp,dim)

        tmp = self.layers[layer].compute(tmp)

        n = len(self.layers)
        for i in range(layer+1, n):
            curr = self.layers[i].diff_chain(tmp, curr)
            tmp = self.layers[i].compute(tmp)
        return curr
    
    def gradient_of_loss(self, inputs:List[float], targets:List[float], layer:int, dim:int)->np.ndarray:
        if layer < 0 or layer >= len(self.layers):
            raise Exception("layer out of range")
        if dim < 0 or dim >= self.layers[layer].weights_count:
            raise Exception("dim out of range")
        predict = self.compute(inputs)
        d_loss_list = self.d_loss(predict,targets)
        # 计算第layer层的对权重的偏导
        grad = self.gradient_at_layer_k_of_dim_n(np.array(inputs),dim,layer)

        result = np.zeros((len(grad[0]),len(grad[0][0])))
        for k in range(len(targets)):
            for i in range(len(grad[0][0])):
                for j in range(len(grad[0])):                
                    result[j][i] += d_loss_list[k] * grad[k][j][i]
        return result
    
    def total_loss(self, inputs:List[List[float]], targets:List[List[float]])->float:
        return sum([self.loss(self.compute(inputs[i]),targets[i]) for i in range(len(inputs))])

    def train_iter(self, inputs:List[List[float]], targets:List[List[float]], learning_rate:float, epoch:int):
        last_loss = self.total_loss(inputs,targets)            
        for n in range(epoch):
            inputs_idx = np.random.permutation(len(inputs))
            for i in range(len(inputs)):
                
                idx = inputs_idx[i]
                for j in range(len(self.layers)):
                    tmp_net = ComplexNet(self.layers,self.loss,self.d_loss)
                    loss = tmp_net.total_loss(inputs,targets)
                    grad = []
                    for k in range(self.layers[j].weights_count):
                        grad.append(self.gradient_of_loss(inputs[idx],targets[idx],j,k))
                    tmp_net.layers[j].update_weights(grad,learning_rate/(0.25+loss))
                    
                    loss = tmp_net.total_loss(inputs,targets)
                    if loss >= last_loss:
                        learning_rate *= math.exp(-0.1*learning_rate)
                    else:
                        self.layers = tmp_net.layers
                        last_loss = loss
            
            print(f'epoch {n} loss: {self.total_loss(inputs,targets)}')
        
def atan(x:List[float])->float:
    return math.atan(x[0])*3

def d_atan(x:List[float])->float:
    return 1 / (1 + x[0] ** 2)*3


def xy(x:List[float])->float:
    return x[0] * x[1]

def d_xy_x(x:List[float])->float:
    return x[1]

def d_xy_y(x:List[float])->float:
    return x[0]

def x_(x:List[float])->float:
    return x[0]

def d_x_(x:List[float])->float:
    return 1

def sin(x:List[float])->float:
    return math.sin(x[0])

def d_sin(x:List[float])->float:
    return math.cos(x[0])

def x2_y2(x:List[float])->float:
    return x[0] ** 2 - x[1] ** 2

def d_x2_y2_x(x:List[float])->float:
    return 2 * x[0]

def d_x2_y2_y(x:List[float])->float:
    return -2 * x[1]

def mse(predict:List[float], targets:List[float])->float:
    return sum([(predict[i] - targets[i]) ** 2 for i in range(len(predict))])
def d_mse(predict:List[float], targets:List[float])->List[float]:
    return [2 * (predict[i] - targets[i]) for i in range(len(predict))]

def huber_loss(predict:List[float], targets:List[float])->float:
    delta = 2
    return sum([0.5 * (predict[i] - targets[i]) ** 2 if abs(predict[i] - targets[i]) <= delta else delta * (abs(predict[i] - targets[i]) - 0.5 * delta) for i in range(len(predict))])
def d_huber_loss(predict:List[float], targets:List[float])->List[float]:
    delta = 2
    return [predict[i] - targets[i] if abs(predict[i] - targets[i]) <= delta else delta * (predict[i] - targets[i]) / abs(predict[i] - targets[i]) for i in range(len(predict))]

def func(x)->float:
    return x*x+max(x,0)*math.sin(x*100)

def test():
    layer1 = ComplexLayer(2,10,sin,[d_sin])
    layer2 = ComplexLayer(10,4,xy,[d_xy_x,d_xy_y])
    layer3 = ComplexLayer(4,4,x2_y2,[d_x2_y2_x,d_x2_y2_y])
    layer4 = ComplexLayer(4,1,x_,[d_x_])

    net = ComplexNet([layer1,layer2,layer3,layer4],mse,d_mse)
    
    x_left = -10
    x_right = 10

    samples = 20
    inputs = []
    targets = []
    for i in range(samples):
        x = x_left + (x_right - x_left) * i / samples
        inputs.append([x,1])
        targets.append([func(x)])
    net.train_iter(inputs,targets,0.5,300)
    x = []
    y = []

    # 尝试外延
    x_left_2 = -10
    x_right_2 = 10
    # real data
    x = np.linspace(x_left_2,x_right_2,1000)
    y = [func(i) for i in x]

    plt.plot(x,y)
    # predict
    x = np.linspace(x_left_2,x_right_2,1000)
    y = [net.compute([i,1])[0] for i in x]
    plt.plot(x,y)
    plt.show()


if __name__ == '__main__':
    test()