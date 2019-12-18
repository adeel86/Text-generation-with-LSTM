# Text generation with LSTM
This deep learning project for text generation with LSTM is written in Jupyter notebooks. 

## Implementing character-level LSTM text generation
Let's put these ideas in practice in a Keras implementation. The first thing I need is a lot of text data that I can use to learn a language model. You could use any sufficiently large text file or set of text files -- Wikipedia, the Lord of the Rings, etc. In this example I will use some of the writings of Nietzsche, the late-19th century German philosopher (translated to English). The language model I will learn will thus be specifically a model of Nietzsche's writing style and topics of choice, rather than a more generic model of the English language.

## Preparing the data
Let's start by downloading the corpus and converting it to lowercase. After executing the code [data_preparing](https://github.com/adeel86/Text-generation-with-LSTM/blob/master/data_preparing.txt). The output will look like this:

```
Corpus length: 600893
```

Next, I will extract partially-overlapping sequences of length maxlen, one-hot encode them and pack them in a 3D Numpy array x of shape (sequences, maxlen, unique_characters). Simultaneously, I prepare a array y containing the corresponding targets: the one-hot encoded characters that come right after each extracted sequence. Code is available at [data_preparing_2](https://github.com/adeel86/Text-generation-with-LSTM/blob/master/data_preparing_2). The output will look likes:

```
Number of sequences: 200278
Unique characters: 57
Vectorization...
```

## Building the network
My network is a single LSTM layer followed by a Dense classifier and softmax over all possible characters. But let me note that recurrent neural networks are not the only way to do sequence data generation; 1D convnets also have proven extremely successful at it in recent times. Since my targets are one-hot encoded, I will use categorical_crossentropy as the loss to train the model. Code is available at [network_building](https://github.com/adeel86/Text-generation-with-LSTM/blob/master/network_building.txt). 

## Training the language model and sampling from it
Given a trained model and a seed text snippet, I generate new text by repeatedly:

* Drawing from the model a probability distribution over the next character given the text available so far
* Reweighting the distribution to a certain "temperature"
* Sampling the next character at random according to the reweighted distribution
* Adding the new character at the end of the available text

This is the code "[training](https://github.com/adeel86/Text-generation-with-LSTM/blob/master/training.txt)" I use to reweight the original probability distribution coming out of the model, and draw a character index from it (the "sampling function"): 

Finally, this is the loop where I repeatedly train and generated text. I start generating text using a range of different temperatures after every epoch. This allows me to see how the generated text evolves as the model starts converging, as well as the impact of temperature in the sampling strategy. You can find the code here [train_and_generate_text](https://github.com/adeel86/Text-generation-with-LSTM/blob/master/train_and_generate_text.txt).

The output file is too lard, therefore I have added output file in the repository [here](https://github.com/adeel86/Text-generation-with-LSTM/blob/master/output.txt). 

## Take aways
* I can generate discrete sequence data by training a model to predict the next tokens(s) given previous tokens.
* In the case of text, such a model is called a "language model" and could be based on either words or characters.
* Sampling the next token requires balance between adhering to what the model judges likely, and introducing randomness.
* One way to handle this is the notion of softmax temperature. Always experiment with different temperatures to find the "right" one.
