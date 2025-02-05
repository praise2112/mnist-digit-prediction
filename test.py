from keras.models import Sequential, load_model
from numpy import zeros, newaxis
import numpy as np


from PIL import Image, ImageFilter


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva



virtualPath = "temp/3.png"

img = Image.open('test/3.jpeg') # image extension *.png,*.jpg

new_width = 28
new_height = 28
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img.save(virtualPath)  # format may what u want ,*.png,*jpg,*.gif


x_ = imageprepare(virtualPath)#file path here
print(len(x_))# mnist IMAGES are 28x28=784 pixels
X_ = np.asarray(x_,  dtype=np.float32)


# mnist_model = load_model("model/keras_mnist.h5")
mnist_model = load_model("results/keras_mnist_3Layer_adam_64BS_128epochs.h5")

# predicted_3 = mnist_model.predict_classes(X_[newaxis, ...])
predicted_3 = mnist_model.predict_proba(X_[newaxis, ...])

# os.remove(virtualPath)

# predicted_classes = mnist_model.predict_classes(X)