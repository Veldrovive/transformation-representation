# Transformation Representation

We assume our dataset has been corrupted by a set of transformations parameterized by a vector $\theta$.

We assume there is some underlying true representation of our data $x'$, that is transformed such that $x=\phi(x', \theta)$.

For each training sample, we know that they belong to a transformation class parameterized by a single index where all in this class share the same $\theta$. We will denote an $x$ from transformation class $m$ as $x^m=\phi(x, \theta_m)$

At test time, we want to learn a function $\hat{y}=f(x^m, \{x_1^m, x_2^m, x_3^m, ...\})$.

**Note:** We may also include the true $y$ for $x_n^m$ for a small variation on this problem such that $\hat{y}=f(x^m, \{(x_1^m, y_1), (x_2^m, y_2), (x_3^m, y_3), ...\})$.

To solve this problem, we propose implicitly learning a useful representation of $\theta$ and $\phi$ while never explicitly regressing these objects.

Define $\gamma(\{x_1^m, x_2^m, x_3^m, ...\})=\gamma(\mathbf{X}^m)$ s.t. $\hat{y} = f(x^m, \gamma(\mathbf{X}^m))$.

**Note:** This may be trained end to end as it is fully differentiable, but $\gamma$ may also be trained using contrastive loss initially and used as a prior. I hypothesize this may lead to worse downstream performance especially on datasets with fewer classes as the representations that are learned may not be suited for the downstream task. Introducing a gradient signal from the downstream task may alleviate this issue, but may also limit the potential size of $\gamma$ or the maximum number of example inputs that can be used during training.

We suggest that $\gamma$ is defined as a convex symmetric function of its parameters. One simple implementation of this would be a function $\gamma(\mathbf{X}^m)=\frac{1}{N}\sum_1^N \eta(x^m_n)$. This class of functions allows us to extend to potentially infinite example inputs and suggests an obvious objective function for $\eta$, a contrastive loss. In this way we can interpret $\gamma(\mathbf{X}^m)$ as being a representation of $\theta^m$.

**Note**: While we could define a new model $\hat{x} = \hat{\phi}^{-1}(x^m, \gamma(\mathbf{X^m}))$ and so map back to canonical $x$, we do not assume that we have access to $x$ so we would also have to learn a representation of $x$. While this is not impossible to do and may have utility, that is not the purpose of this initial exploration as we are more interested in learning some parameterization of the transformation for downstream tasks. I would be interested in this in the future, however, as it allows us to formulate this entire problem as an instance of representation learning. It could be performed also using a contrastive loss at the output of $\hat{\phi}^{-1}$ either though a high cosine similarity across modalities as with CLIP or LiT, or a standard contrastive objective to examples known to share an underlying connection through some shared task $y$.

For an example of where this may be applicable, imagine the set of all cameras with all their varying parameters (for an extreme case we could even introduce wildly divergent cameras such as some function of event based signals or those scrambled by a filter). We now want to perform a classification task with any of these cameras. We get a few pictures that have been taken as examples and now need to classify as cats or dogs.

In order to test this method, we first modify the standard MNIST task by defining a set of transformations to the images.

We then increase the complexity by performing classification of imagenet though transformations.

For a test of effectiveness, we also train a normal network without using any example inputs with a similar number of parameters varying the severity and variation in transformations.

### Personal Notes:
I think this proof of concept serves better if it is just plain contrastive loss on the transformation without any gamma function. Once we know how effective that is we can then introduce combination rules that allow us to take into account multiple examples. We can explore plain combinations with means or other symmetric functions.

We could also explore more advanced combination rules such as attention.

We can also explore this separately in something like learning embeddings of videos. Or better use a masking method to remove most of the information in the image and then use combination to recover it to validate the method which can then be applied to stuff like video scene representation or more complex things like transformation representation.

For this task it would be interesting to explore where to exchange information between the representations. Like in a convnet would it be useful to have an interem layer before hitting the representation where information can be exchanged?

I guess this task is something like multi-instance representation learning or fusion representation learning?

There's also the idea of grounding the system more in bayesian statistics by treating the representation as a transformation from a uniform to unknown distribution by sampling the space and treating the result as a system we can perform filtering on. Not exactly sure of the training process. Maybe generate a mixture model then have the contrastive objective approximate a KL divergence maximizing that of the positive example and minimizing the negative example. I guess the number of samples in the mixture model would be a hyperparameter. This is most useful if sometimes the mixture model can well approximated by completely disjoint set of distributions as otherwise we could just view this as manually defining part of the latent space manifold. Although in this case we are also trying to get the networks to quantify its uncertainty through the use of predicting the standard deviation of the transformed guassians so maybe even then it couldn't be approximated by a different latent space geometry.