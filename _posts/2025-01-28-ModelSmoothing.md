---
title: "Model Smoothing to Prevent Zero-Probabilities in Probabilistic Language Models"
excerpt: "Notes on model smoothing in machine learning"

writer: Ji Kim
categories:
    - Deep Learning
tags:
    - [Model Smoothing]

toc: true
toc_sticky: true

math: true
date: 2025-01-28
last_modified_at: 2025-01-28
---

## Introduction
Probabilistic language models is trained based on the statistics of the training corpus. However, no matter how large the training corpus is, there is always a possibility that the model encounters an **Out-of-Vocabulary (OOV)** word during validation or test phase. This indicates that the model has an inherent **data sparsity** problem.

To address this issue, *model smoothing* techniques are used to prevent the model from assigning zero probability to any OOV words. It is a process of flattening the probability distribution of the model to prevent extreme values (recall negative infinity assigned to OOV words from the previous post).

## Laplace Smoothing (Additive Smoothing)
[Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) is the simplest model smoothing technique. It is a process of **adding a small constant** value to the count of each word in the training corpus. This constant value is usually set to 1, which is why it is called *additive smoothing*.

Probability of a word $w_i$: 

$$
P(w_i) = \frac{C(w_i)}{N}
$$

After Laplace smoothing, the probability of a word $w_i$ is:

$$
P(w_i) = \frac{C(w_i) + \alpha}{N + \alpha V}
$$

Where:
- $\alpha$: the smoothing parameter.
- $N$: the total number of words in the training corpus.
- $V$: the number of unique words in the training corpus.

## Katz Back-off Smoothing

[Back-off smoothing](https://en.wikipedia.org/wiki/Katz%27s_back-off_model) is a technique that recursively backs off to a lower-order model when the higher-order n-gram model is not available. For example, let's say we have unigram, bigram, and trigram models trained from the identical training corpus. If the probability is unassignabled by the higher-order model, the probability is assigned by the lower-order model.

$$
P(w_i|w_{i-1},\ldots,w_{i-n+1}) = 
$$

$$
\begin{cases} 
\lambda(w_{i-1},\ldots,w_{i-n+1}) \times P(w_i|w_{i-1},\ldots,w_{i-n+1}), & \text{if count}(w_{i-1},\ldots,w_{i-n+1},w_i) > 0 \\
\alpha(w_{i-1},\ldots,w_{i-n+1}) \times P(w_i|w_{i-1},\ldots,w_{i-n+2}), & \text{otherwise}
\end{cases}
$$

Where:
- $\lambda(w_{i-1},\ldots,w_{i-n+1})$: the back-off weight.
- $\alpha(w_{i-1},\ldots,w_{i-n+1})$: the discounting factor.
- $P(w_i \mid w_{i-1},\ldots,w_{i-n+1})$: the probability assigned by the higher-order model.
- $P(w_i \mid w_{i-1},\ldots,w_{i-n+2})$: the probability assigned by the lower-order model.

Therefore: 
- If the trigram exists in the training corpus:
$$
\lambda(w_{i-1},\ldots,w_{i-n+1}) \times P(w_i|w_{i-1},\ldots,w_{i-n+1})
$$

- If the trigram does not exist in the training corpus:
$$
\alpha(w_{i-1},\ldots,w_{i-n+1}) \times P(w_i|w_{i-1},\ldots,w_{i-n+2})
$$

### Why is Discounting Necessary?

- Maintains the Probability Mass Valid
    - Simply grabbing the lower-level n-gram probability could overestimate the probability of the higher-order n-gram, causing the probability mass to exceed 1.   
- Accounts for Unseen N-grams
    - Some n-grams may not be present in the training corpus, by discounting, we reserve some probability mass for such missing higher-order n-grams.
- Prevents Double-counting
    - If we were to use the higher-order n-gram probability directly, we would be double-counting the probability mass.

### Example

Let's say we are assigning a probability to a trigram `"The cat sleeps"` in a trigram model.

The trigram does not exist in the training corpus, so we need to use the back-off model where:

$$
P(sleeps| \text{cat}) = 0.5
$$

$$
P(sleeps| \text{the}, \text{cat}) = \lambda_1 \cdot P(sleeps| \text{cat}) = 0.4 \times 0.5 = 0.2
$$

If `"cat sleeps"` does not exist in the training corpus, we further back-off to the unigram model:

$$
P(sleeps| \text{the, cat}) = \lambda_1 \cdot \lambda_2 \cdot P(sleeps) = 0.4 \times 0.6 \times 0.1 = 0.024
$$

## Conclusion

In this post, we have discussed two popular model smoothing techniques: **Laplace Smoothing** and **Katz Back-off Smoothing**. Both techniques help address the data sparsity problem in language models by preventing zero probabilities for unseen words or n-grams. While Laplace smoothing offers a simple solution by adding a constant to all counts, Katz Back-off provides a more sophisticated approach by leveraging information from lower-order n-grams.

In future posts, we will explore other advanced smoothing techniques such as **Interpolation** and **Kneser-Ney Smoothing**, which are widely used in modern language modeling applications due to their superior performance in handling rare and unseen events.

## Resources

- [Language Model (2) Smoothing](https://heiwais25.github.io/nlp/2019/10/06/Language-model-2/)
- [katzs-back-off-model-in-language-modeling](https://www.geeksforgeeks.org/katzs-back-off-model-in-language-modeling/)