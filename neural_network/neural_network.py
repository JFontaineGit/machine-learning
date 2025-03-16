import numpy as np

# Red Neuronal Multicapa (evitamos tener que definir cada capa manualmente, más práctico)
class MultiLayerNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=1.9):
        self.layers = []
        self.learning_rate = learning_rate

        # Capas ocultas
        previous_size = input_size
        for size in hidden_sizes:
            self.layers.append({
                'W': np.random.randn(previous_size, size),
                'b': np.zeros((1, size)),
            })
            previous_size = size

        # Capas de salida
        self.layers.append({
            'W': np.random.randn(previous_size, output_size),
            'b': np.zeros((1, output_size)),
        })
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, x):
        a = x
        activations = [a]  # Activaciones de cada capa

        for layer in self.layers:
            z = np.dot(a, layer['W']) + layer['b']
            a = self.sigmoid(z)
            activations.append(a)

        return activations

    def backward_propagation(self, x, y, activations):
        m = x.shape[0]  # Número de ejemplos (muestras)
        error = activations[-1] - y

        # Propagación hacia atrás
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = activations[i]
            dz = error * self.sigmoid_derivative(activations[i + 1])
            dw = np.dot(a_prev.T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            # Actualización de pesos y sesgos
            layer['W'] -= self.learning_rate * dw
            layer['b'] -= self.learning_rate * db

            # Propagar el error hacia la capa anterior
            error = np.dot(dz, layer['W'].T)
    
    def train(self, x, y, epochs=10000):
        for epoch in range(epochs):
            activations = self.forward_propagation(x)
            self.backward_propagation(x, y, activations)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(activations[-1] - y))
                print(f"Epoch {epoch} - Loss: {loss}")
    
    def predict(self, x):
        activations = self.forward_propagation(x)
        return np.round(activations[-1]).astype(int)
    
