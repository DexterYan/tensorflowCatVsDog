### 1. What is this
It is a classifer of reconizing the cat or dog based on convolutional neural net, using tensorflow

### 2. Installation
First, you have to follow the [Tensorflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#download-and-setup) Setup to install a tensorflow environment.
Second, install these library by pip
- [tflearn](https://github.com/tflearn/tflearn)
- [numpy](http://www.numpy.org/)
- [scikit-image](http://scikit-image.org/)
- [scipy](https://www.scipy.org/)
- [Python Imaging Library](http://www.pythonware.com/products/pil/)
- [matplotlib](http://matplotlib.org/)

Third, download the [dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from  kaggle. Put the train image into train folder

### 3. Usage
Active the tensorflow
-
To train the neural network, run
```
python classifier.py
```

- To do the predicate the image, run
```
python predict.py
```
make sure you have put some image in the test folder, by default, I have train a neurial work for you, you can directly run predicate

### 4. Folder Struture
- test/ (you can put your image for testing)
- train/ (dataset from kraggle.com)
- tmp/ (log file)

### 5. Something missing? Fork!
fork this repo; send a pull request!; My email yanshaocong@gmail.com

### MIT

Copyright (c) 2016 Dexter Yan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
