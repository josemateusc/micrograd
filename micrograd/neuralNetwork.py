import random
from micrograd.engine import Value
import numpy as np


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, act="tanh"):
        self.w = [
            Value(random.uniform(-1, 1)) for _ in range(nin)
        ]  # weight for each input
        self.b = Value(random.uniform(-1, 1))  # bias
        self.act = act

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # activation
        out = act.tanh() if self.act == "tanh" else act.relu()
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)}, act={self.act})"


class Layer(Module):

    def __init__(
        self, nin, nout, act="tanh"
    ):  # dimension of the neurons and number of neurons
        self.neurons = [Neuron(nin, act=act) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #   ps = neuron.parameters()
        #   params.extend(ps)
        # return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(
        self, nin, nouts, act="tanh"
    ):  # number of inputs and size of the layers
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], act=act) for i in range(len(nouts))
        ]  # set the size of input and output of each layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def loss(self, X, y, loss_type="svm", batch_size=None):
        # inline DataLoader :)
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]
        inputs = [list(map(Value, xrow)) for xrow in Xb]

        # forward the model to get scores
        scores = list(map(self, inputs))

        if loss_type == "svm":
            # svm "max-margin" loss
            losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
        elif loss_type == "mse":
            # Mean Squared Error loss
            losses = [(scorei - yi) ** 2 for yi, scorei in zip(yb, scores)]
        else:
            raise ValueError("Unsupported loss type")

        data_loss = sum(losses) * (1.0 / len(losses))

        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in self.parameters()))
        total_loss = data_loss + reg_loss

        # also get accuracy
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    def train(self, X, y, epochs=100, patience=10, loss_type="svm"):
        best_loss = float("inf")
        patience_counter = 0
        best_parameters = None

        for k in range(epochs):
            # forward
            total_loss, acc = self.loss(X, y, loss_type=loss_type)

            # backward
            self.zero_grad()
            total_loss.backward()

            # update (sgd)
            learning_rate = 1.0 - 0.9 * k / epochs
            for p in self.parameters():
                p.data -= learning_rate * p.grad

            if k % 1 == 0:
                print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

            # early stopping and saving best parameters
            if total_loss.data < best_loss:
                best_loss = total_loss.data
                patience_counter = 0
                best_parameters = [p.data for p in self.parameters()]
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Stopping early at epoch {k} with best loss {best_loss}")
                break

        # Restore best parameters
        if best_parameters:
            for i, p in enumerate(self.parameters()):
                p.data = best_parameters[i]
