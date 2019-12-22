from tqdm.auto import tqdm
import alexnet_model.alexnet
from alexnet_model.alexnet import AlexnetModel
import numpy as np
from PIL import Image
from alexnet_model.classes import classes
import tensorflow as tf
from _datetime import datetime
from enums import NeuronChoice

def getSummaryWriter(modelName):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_path = f'logs/{modelName}/' + current_time
    print("run: tensorboard --logdir ./" + base_path + " --port 6006")
    return tf.summary.create_file_writer(base_path)


def getImage(imageName,directory="./alexnet_weights/", normelize = None):
    I = Image.open(directory + imageName).resize([224, 224])
    I = np.asarray(I).astype(np.float32)
    I = I[:, :, :3]

    I = np.flip(I, 2)  # BGR
    I = I - [[[104.00698793, 116.66876762, 122.67891434]]]  # subtract mean - whitening
    # I = I - np.mean(I,axis=(0,1),keepdims=True)
    I = np.reshape(I, (1,) + I.shape)
    return I if not normelize else normelize(I)


def getModel(img_name, img_dir, weight_dir) -> (AlexnetModel, np.ndarray):
    model = AlexnetModel()
    I = getImage(img_name, img_dir)
    model(I)
    model.setAlexnetWeights(weight_dir)
    return model, I

def getLossFunction(normalizition_lambda=1e-3, norm= None):
    if norm:
        normelizer = norm
    else:
        normelizer = tf.norm

    @tf.function
    def loss_object(neuron, I ):
        return neuron - normalizition_lambda * (normelizer(I) ** 2)

    return loss_object



def get_train_step(model: AlexnetModel, I, loss_object, neuronChoise : NeuronChoice ):
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            prediction, outputs = model(I)
            wanted_layer = outputs[neuronChoise.layer]
            layer_shape = wanted_layer.shape
            if len(layer_shape) == 2:  # affine layer:
                neuron = wanted_layer[0][neuronChoise.index]
                Sc = loss_object(neuron, I)
            else:  # conv layer
                neuron = wanted_layer[0]
                neuron_pixel = neuron[neuronChoise.row,neuronChoise.col, neuronChoise.filter]
                Sc = loss_object(neuron_pixel, I)
            actual_loss = -Sc
            gradients = tape.gradient(actual_loss, [I])
            optimizer.apply_gradients(zip(gradients, [I]))


    # return train_step, train_loss, train_accuracy
    return train_step, loss_object


def train():
    # import shutil # Uncomment if you want to clear the folder
    # shutil.rmtree("./logs/Q1-I")
    model, I = getModel("poodle.png", "./alexnet_weights/", "./alexnet_weights/")
    I_v = tf.Variable(initial_value=tf.random.normal((1, 224, 224, 3)), trainable=True)
    I_v.initialized_value()

    loss_object = getLossFunction()
    neuronChoice = NeuronChoice(layer = "conv2", filter = 78, row = 5, col = 5,)
    summaryWriter = getSummaryWriter("Q1-I")
    train_step, train_loss = get_train_step(model, I_v, loss_object, neuronChoice)
    iter_count = 5000
    for i in tqdm(range(1, iter_count + 1)):
        train_step()
        if i % 100 == 0:
            plot_i = I_v - tf.reduce_min(I_v)
            plot_i = plot_i / tf.reduce_max(I_v)
            with summaryWriter.as_default():
                tf.summary.image(neuronChoice.layer, plot_i, step=i)


    summaryWriter.close()

def main():
    # Create an instance of the model
    model, I = getModel("poodle.png", "./alexnet_weights/", "./alexnet_weights/")
    c, _ = model(I)
    top_ind = np.argmax(c)
    print("Top1: %d, %s" % (top_ind, classes[top_ind]))


if __name__ == "__main__":
    train()
    # main()
    # load_model("alexnet_weights")
