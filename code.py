import numpy as np
import tensorflow as tf


def load_coffee_data():

    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          
    X[:,0] = X[:,0] * (285-150) + 150  
    Y = np.zeros(len(X))
    
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))


X,Y = load_coffee_data()

norm_l = tf.keras.layers.Normalization(axis = -1)  #normalized the input
norm_l.adapt(X)    #learn mean and variance

model = tf.keras.Sequential([
    norm_l,
    tf.keras.layers.Dense(5, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy'
)

model.fit(X,Y, epochs = 100)

predictions = model.predict(X)

W1, b1 = model.layers[1].get_weights()
W2, b2 = model.layers[2].get_weights()
W3, b3 = model.layers[3].get_weights()

print(f"Weight matrix of layer 1:\n{W1}\nBias vector of layer 1{b1}\n\n\n")
print(f"Weight matrix of layer 2:\n{W2}\nBias vector of layer 2{b2}\n\n\n")
print(f"Weight matrix of layer 3:\n{W3}\nBias vector of layer 3{b3}\n\n\n")

labels = (predictions >=0.5).astype(int)  #converting the predictions to binary labels

print(f"{labels[:10]}\n\n\n") #prints the first 10 predictions from X

a = int(input("enter temperature in degree Celcius "))
b = int(input("enter time in minutes "))

new_coffee = np.array([[a,b]])
new_prediction = model.predict(new_coffee)
new_label = (new_prediction >= 0.5).astype(int)
if new_label[0][0] == 1:
    print("The time and temperature are favourable and can be used to roast coffee beans.")
else:
    print("The time and temperature are not favourable.")


