# 1. What is word2vec?

Word vector, or word2vec, simply put, is the conversion of commonly used characters, letters, and so on into numbers. This is because, for computers, they can only understand binary numbers. However, for humans, decimal numbers are easier to understand than binary numbers. Therefore, people first convert words into decimal numbers.

For computers, the conversion of word vectors is a specific data processing method in the field of natural language processing (NLP). In the computer vision (CV) field, images themselves are stored as numbers in computers, and these numbers already contain certain information. At the same time, there are already some relationships between different groups of numbers. For example, if two images are both pictures of the sea, then both images will have a predominance of blue, and the proportion of the letter "B" in the RGB numbers of the two images will be relatively large. Of course, there are other feature connections, but because humans are not sensitive to numbers, some information cannot be directly discovered by people.

The quality of word vectors directly affects the subsequent processing of NLP, such as machine translation, image understanding, and so on. Without high-quality word vectors, the quality of machine translation cannot be effectively improved.

# 2. Two main word2vec methods.
## 2.0 An example
Imagine you're teaching a computer to understand words like humans do. We know words not just by their letters, but also by the words they are usually surrounded by.

Now, let's take the word "king." Humans often see "king" in contexts like "queen," "throne," or "kingdom." Word2Vec does something similar. It learns to represent words as points in a multi-dimensional space based on the words that often appear around them.

There are two methods Word2Vec does this:

## 2.1 CBOW
- CBOW predicts a target word based on its context, which consists of surrounding words. It aims to predict a target word given the context words within a specific window.
CBOW learns to predict a target word (like "king") based on the words around it. So, if you give it the words "queen," "throne," and "kingdom," it will try to guess the target word "king."
## 2.2 Skip-gram
- Skip-gram, on the other hand, predicts context words given a target word. It tries to maximize the likelihood of predicting context words based on the input target word.
Skip-gram is the opposite. It takes a word (like "king") and tries to predict the words around it. So, if you give it the word "king," it will try to guess "queen," "throne," and "kingdom."