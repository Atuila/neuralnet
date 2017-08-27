from PIL import Image
import math
import random

sizex = 100
sizey = 100
nbIteration = 100000
nbImage = 20

def sigmoide(x):
    # Empeche les overflow
    x = min(200, x)
    x = max(-200, x)

    return 1 / (1 + math.exp(-x))

class Neuron:
    def __init__(self, weights, isOutput):
        self.weights = weights
        self.isOutput = isOutput

    # Somme pondérée des inputs par les poids
    def activate(self, entry):
        activation = 0
        for i, value in enumerate(entry):
            activation += value * self.weights[i]

        return sigmoide(activation)

# Génère les couches de neurones et leurs poids en fonction des couches demandées
def generateNetwork(neuronSizes):
    network = []
    for i in range(len(neuronSizes) - 1):
        network.append([Neuron(
            # Poids aléatoires
            [random.random() for loop in range(neuronSizes[i])],
            i == len(neuronSizes) - 2 # Si neurone d'output
        ) for loop in range(neuronSizes[i + 1])])

    return network

# Reseau de neurone
network = generateNetwork([7, 15, 15, 3])

# Calcule un pixel pour (x, y)
def calcOutput(x, y):
    # Paramètres d'entrée du réseau
    x = (x - (sizex/2)) / (sizex/2)
    y = (y - (sizey/2)) / (sizey/2)
    values = [[x, y, x ** 2, y ** 2, x * y, x + y, 1.]]

    # Parcoure chaque couche 1 à 1
    for layer in network:
        # Calcule les valeurs retournées par chaque neurone de la couche en fonction
        # des valeurs retournées par la couche précédente
        activations = []
        for neuron in layer:
            activations.append(neuron.activate(values[-1]))

        values.append(activations)

    return values

# Transforme la valeur retournée par le réseau en couleur
def calcColor(x, y):
    return tuple(list(map(lambda z: int(255 * z), calcOutput(x, y)[-1])))

# Entraine l'algorithme par backpropagation
def train(x, y, expected):
    # Pas d'apprentissage
    l_rate = 1

    # Valeurs retournées par chaque couche
    networkOutput = calcOutput(x, y)

    # Traverse le réseau de la fin au début
    for i in reversed(range(len(network))):
        layer = network[i]
        for j, neuron in enumerate(layer):
            output = networkOutput[i + 1][j]
            # Si neurone d'output
            if neuron.isOutput:
                # Dérivée de la fonction erreur + sigmoide
                neuron.derivative = output * (output - expected[j] / 255) * (1 - output)
            else:
                derivative = 0
                # Dérivée des neurones suivants x le poid entre les 2 neurones
                for k, rneuron in enumerate(network[i + 1]):
                    derivative += rneuron.derivative * rneuron.weights[j]

                neuron.derivative = derivative * output * (1 - output)

            # Calcul des mises à jour des poids
            for k, weight in enumerate(neuron.weights):
                neuron.weights[k] -= l_rate * networkOutput[i][k] * neuron.derivative

# Image à apprendre
img = Image.open('Image3.png')
pix = img.load()

# Image générée
nimg = Image.new("RGB", (sizex, sizey))
npix = nimg.load()
for x in range(sizex):
    for y in range(sizey):
        npix[x, y] = calcColor(x, y)

nimg.show()

# Entrainement, on prend un pixel au hasard
for loop in range(nbIteration):
    if sizex == sizey:
        x, y = (int(sizex * random.random()) for loop in range(2))
    else:
        x = int(sizex * random.random())
        y = int(sizey * random.random())
    train(x, y, pix[x, y])

    if loop % int(nbIteration / nbImage) == 0:
        for x in range(sizex):
            for y in range(sizey):
                npix[x, y] = calcColor(x, y)
        numImage = str(int(random.random() * 200000))
        nimg.save("C:/Users/Matis/Desktop/neuralnet/ImageGen/Image " + numImage + ".bmp")
        print(loop)

# Image générée après entrainement
for x in range(sizex):
    for y in range(sizey):
        npix[x, y] = calcColor(x, y)

nimg.show()
