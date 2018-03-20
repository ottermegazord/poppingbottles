
# Importing the Keras libraries and packages

from keras.models import load_model
from keras.utils import plot_model
import numpy as np
from keras.preprocessing import image

class Cell():

    def __init__(self, img, model):
        self.img = img
        self.model = load_model(model)

    def predict(self):
        test_image = self.img
        # test_image = image.load_img('green3.jpg', target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = self.model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'red'
            print("it's a cell")
        else:
            prediction = 'green'
            print("yup, it's a bead.")

        # print(result)

#
# cell = Cell('green.jpg', "cellClassifier.h5")
# cell.predict()




# classifier = Sequential()
# model = load_model("cellClassifier.h5")
# plot_model(model, to_file='model.png')
#
# test_image = image.load_img('green3.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
#
# result = model.predict(test_image)
#
# print(result);

# if result[0][0] == 1:
#     prediction = 'red'
#     print("it's a cell")
# else:
#     prediction = 'green'
#     print("yup, it's a bead.")
