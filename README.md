# Chatbot for Traveloka

Chatbot model deployment using Flask framework

## Project Domain

The chatbot is a computer program that simulates and processes human interactions. The chatbot can be a solution to help improve service to customers at a very low cost. The problem we want to tackle is to improve customer service by creating a chatbot to answer questions that are often asked by customers. We also want to make customers able to explore more about the products on Traveloka through hotel and restaurants recommendation on our chatbot. Machine learning has a critical contribution to the chatbot model development. We developed the model starts from creating the JSON dataset from scratch, doing data preparation, and then building the model using Tensorflow for text classification. For model evaluation, we show the accuracy and loss through the graph in each epoch. Lastly, we deploy and test the model on a local server which will be continued by the CC team for deployment in the cloud services.

## Tech and tools

- Tensorflow
- Scikit-learn
- NLTK
- Numpy
- Pandas

## How to replicate

- Fork the repository
- Install the requirements using :
  pip install requirements.txt
- Run app.py
- Make post request using postman or similliar software

## RNN Model

### Model Summary

We make the model that can classify 13 tags that including:

- greeting
- general payment info
- booking flights
- booking hotels
- hotel reschedule
- flight reschedule
- hotel refund
- boarding ticket refund
- travel credit
- call center
- hotel recommendation
- restaurant recommendation
- closing

![Model Summary](assets\images\model_summary.png)

We make the model with Tensorflow. The layers are starts from the Embedding layer using GloVe 100-Dimensional Word Vectors followed by two layers of LSTM inside the Bidirectional layers. Then we add some dense layers before the output layer. To reduce the overfitting, we add some dropout between the dense layers and regularizers L1 and L2 inside the Dense layers.

### Model Result

As we can see from the graph below, accuracy keeps on increasing as well as the loss keeps decreasing for 150 epochs. The final result is that we got 0.99 in training accuracy and 0.97 in validation accuracy

![Accuracy Loss Graph](assets\images\loss_accuracy_graph.png)
