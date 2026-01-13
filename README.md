The dataset is the script of a well known movie.
Using Tensorflow, I built this autoregressive language processing model that takes an input and predicts what comes next, character by character.
It predicts the next character on the basis of its conditional probability given the previous sequence of characters.
The prediction is then iteratively sent in as the input to continue the predictions.
The architecture used is LSTM(Long Short Term Memory), a kind of recurrent neural network that specializes in dealing with sequential data avoiding traditional issues(vanishing gradient).
This project was aimed at the objective of learning some important concepts of NLP(Natural Language Processing) such as autoregressive sequence modeling, next-token prediction, LSTM-based language models, text generation,etc.
