# Evaluation_Gutenberg

This project aims at comparing the topology of two spaces.
One is the ground truth, encoding the genre labelling of texts of a corpora.
The second is the representation of texts, based of features that should be linked with the textual genre.

We compare here, to which extend this ground truth space, and the representations share some topological properties.

Moreover, we try, using the RRelief algorithm, ElasticNet and some Neural Network, to transform the models to make them match better the ground truth space.

---
Let's suppose that we have a collection of elements. Each element is represented in two different vector representations. The question is how to make one vector representation more similar to the other.

We consider two different ways to make a topology. First, we consider that to induced topologies on the two vector representations are the same if the set of distance between every points, in the two space, is the same. Thus, we would look for a transformation of the vector space $V_1$ to a vector space $V_1'$ which would give a high correlation of the distances between points.
The point is that if we have two representation, and that we use the same metric function, the induced topology would depend only on the vector representation, thus we have to tweak it in order to push the to space to have a more similar topology.

The second approach works on the notion of neighbors.
We still use the induced topology on the two spaces, but we consider that neighbors should be closer that non neighbors in the embedding space.
Thus the second approach also work to change the vector representations, but the data used to compute the best representation aren't the same.

---

For the first method, we sample random points in our dataset, we embed them into the new vector space and then compute their distances.
We compute the loss as follow
$$
  \mathcal{Loss} = d_{v1'}(X_i,X_j) -  
$$
