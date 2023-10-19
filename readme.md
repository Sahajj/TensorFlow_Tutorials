# Tensorflow and Deep learning

## Machine learning
- It is turning things (data) into numbers and finding patterns in those numbers.

- Finding patterns --> The computer does this part How? 
code & math we're going to be writing the code 

# Machine learning vs Deep learning 

AI --> ML --> Deep Learning
- What's the Difference between Traditional Programming and Machine Learning?

- with Traditional programming we start with <br> Inputs --> Rules --> output (Like making soup)

- with the ML we start with <br> Showing inputs --> telling about Ideal Outputs --> Rules  (The Algo will figure out the process from which we can get from input to output.) 


## Why use Machine Learning (or Deep learning)

- Why not? Maybe
- Better reason: For a complex problem, can you think of all the rules? (example driving a car?) probably not.

- When you should not use ML?
- If you can build a simple rule-based system that doesn't require machine learning, do that.<br>-- A wise software engineer... (actually rule 1 of google's Machine Learning Handbook)

## What is Deep learning is Good for?

- **Problems with long lists of rules** -- When We traditional approach fails, machine learning/ deep learning may be helpful. 
- **Continually changing environments**  -- Deep learning can adapt ('learn') to new scenarios.
- **Discovering insights within large collections of data** --  can you imagine trying to handcraft rules for what 101 different kinds of food look like?

## What is Deep learning Not Good for ? "typically"

- **When you need expandability** --  the pattern learned by a deep learning model is typically uninterpretable by a human.

- **When the traditional approach is a better option** -- if you can accomplish what you need with a simple rule-based system.

- **When errors are unacceptable** -- since the outputs of the deep learning model aren't always predictable.

- **When you don't have much data**-- deep learning model usually requires a fairly large amount of data to produce great results. (though we'll see how to get great results without huge amounts of data.)

# Machine learning vs Deep learning 

- ML works with more Structured data and DL works better with unstructured data.

| Machine Learning  |       Deep learning |
| --------          |              ------ |
|  Random Forest    | Neural Networks     |
|  Naive Bayes      |  Fully connected neural networks              |
| Nearest neighbor  |   Convolutional neural networks             |
|  Support vector machine |  Recurrent neural networks             |
| many more       |  Transformers and many more       |
| After DL these are aka shallow Algorithms  |         |

- we will be focusing on 
    1. Neural Networks
    2. Fully connected neural networks 
    3. convolutional neural network
    4. recurrent neural network


## What is Neural Networks?

- **inputs** --> **numerical encoding**(changing data into numbers) -->  **Learns representation (patters/features/weights)** (choose the appropriate neural network for your problems) --> **representation outputs** --> **outputs** (a human cna understand this)


## Anatomy of Neural Network

- Input layers (data goes here)
- Hidden layer(s) (learns pattern in data)
- output layer (output representation or prediction probabilities.)

- Note: the pattern is an arbitrary term, you'll often hear "embedding", "weights", "feature representation", and "feature vectors" all referring to similar things.

## types of learning 

1. Supervised 
2. semi-supervised 
3.  unsupervised 
4. transfer learning/reinforcement learning.

## What is deep learning actually used for?

1. recommendation
2. translation 
3. speech recognition
4. computer vision 
5. Natural Language Processing (NLP)

- translation & speech recognition are more *sequence to sequence* *
- Natural Language Processing (NLP) *classification /regression*

## TensorFlow
- an end-to-end platform for machine learning 
- write fast and deep learning code in Python/other accessible languages(able to run on GPU/TPU)
- able to access many pre-build learning models (TensorFlow Hub)
- whole Stack: pre-process data, model data, deploy the model in your application.
- originally designed and used in-house by Google (now  open-source).

### Why TensorFlow
 - Easy model building (A lot of this) 
 - Robust Ml production 
 - Powerful experimentation for research (some of this)

 - go to https://tensorflow.org

- What is GPU --> Graphical Processing Unit
<br>TPU --> Tensor Processing Unit

## What is tensor?

 - The numerical way to represent patterns. 

## We are going to cover
 - TensorFlow basics and Fundamentals 
 - Processing Data (getting it into tensors)
 - Building and Using Pre-trained deep learning models 
 - Fitting a model to the learning patterns)
 - making predictions with a model(using patterns)
 - Evaluating mode predictions
 - Saving and loading models
 - Using a trained model to make predictions on **custom data**.

### How  to approach the course?

- Write code(lot's of it, follow along, let's make mistakes together)
    - Motto 
         1. "If in doubt, run the code" 
-  Explore and Experiment
    - Motto 
        1. "Experiment, Experiment, Experiment"
        2. "Visualize, Visualize, Visualize " (recreate things in ways you can understand) 



