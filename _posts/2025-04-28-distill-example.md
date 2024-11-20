---
layout: distill
title: Diffusion Models Meet Flow Matching
description: Flow matching and diffusion models have become prominent frameworks in generative modeling, with flow matching drawing growing attention recently. Despite being seemingly similar, there is general confusion in the community about their exact connection. In this blog post, we systematically analyze and connect the two frameworks, covering every facet in their training and sampling procedures. With all investigation we verify a claim affirmatively, i.e., diffusion model and Gaussian flow matching are essentially the same. One should feel comfortable using the two frameworks interchangeably. 
date: 2025-11-12
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Overview
  - name: Training
    subsections:
    - name: Network output
    - name: Noise schedule
    - name: Weighting function
  - name: Sampling and Straightness Misnomer

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

{% include figure.html path="assets/img/2025-04-28-distill-example/twotrees.jpg" class="img-fluid" %}


Diffusion models and flow matching have emerged as powerful frameworks in generative modeling. In particular, flow matching has gained inreasing popularity recently, due to its simplicity in formulation and "straightness" in the sampling trajectories. A common question one hears nowadays is: 


<!-- > Does this diffusion technique also work with Gaussian flow matching? -->
<p align="center"><i>"Does this diffusion technique also work with Gaussian flow matching?"</i></p>

Clearly, there is confusion in the field. After all, a diffusion model and a Gaussian flow matching are essentially equavalent. Therefore, the answer is yes, trivially.

In this blog post, we will walk through the frameworks of diffusion model and flow matching systematically from the practical perspective. We mainly focus on Gaussian flow matching with the optimal transport flow path <d-cite key="lipman2022flow"></d-cite>, the dominant version of flow matching adopted by the field of media generation. Other closely related frameworks include rectified flow <d-cite key="liu2022flow"></d-cite> and stochastic interpolant <d-cite key="albergo2023stochastic"></d-cite>. Our purpose is not to downweigh the importance of either framework. In fact, it is interesting to see that two frameworks derived from distinct theoretical perspectives lead to the same algorithm in practice. Rather, we hope to make practitioner feel comfortable to use the two frameworks interchangeably, understand the actual degrees of freedom we have when tuning the algorithm (no matter how we name it), and what design choices actually do not matter.


## Overview

We start by recalling the two frameworks (diffusion models and flow matching) and compare them from a high level. 
<!-- We highlight the free parameters in each framework and how they relate to each other. 
In particular, there exists explicit mappings to define a diffusion model from a flow matching model and vice-versa. This overview does not dive into the training of such models, i.e., we assume that all the learnable quantities have been adequatly optimized. We also do not discuss the different sampling techniques used at inference. Both the training and the inference will be discussed in further sections.  -->


### Diffusion models

A diffusion process gradually destroys an observed data $$ \bf{x} $$ over time $$t$$, by mixing the data with Gaussian noise:
$$
\begin{equation}
{\bf z}_t = \alpha_t {\bf x} + \sigma_t {\boldsymbol \epsilon}, \;\mathrm{where} \; {\boldsymbol \epsilon} \sim \mathcal{N}(0, {\bf I}).
\label{eq:forward}
\end{equation}
$$
$$\alpha_t$$ and $$\sigma_t$$ define the **noise schedule**. A useful notation is the log signal-to-noise ratio $$\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$$, which monotonically decreases as $$t$$ increases from $$0$$ to $$1$$ (i.e., goes from clean data to Gaussian noise). 

To generate new samples, we can "reverse" the forward process gradually: Initialize the sample from Gaussian at the highest noise level. Given the sample $${\bf z}_t$$ at time step $$t$$, we predict what the clean sample might look like with a neural network $$\hat{\bf x} = \hat{\bf x}({\bf z}_t; t)$$, and then we project it back to a lower noise level with the same forward transformation:

$$
\begin{eqnarray}
{\bf z}_{t - \Delta t} &=& \alpha_{t - \Delta t} \hat{\bf x} + \sigma_{t - \Delta t} \hat{\boldsymbol \epsilon},\\
\end{eqnarray}
$$
where $$\hat{\boldsymbol \epsilon} = ({\bf z}_t - \alpha_t \hat{\bf x}) / \sigma_t$$. We keep alternating between predicting the clean data, and projecting it back to a lower noise level until we get the clean sample.
This is the DDIM sampler <d-cite key="song2020denoising"></d-cite>. The randomness of samples only comes from the initial Gaussian sample, and the entire reverse process is deterministic. In the sampling session we will discuss other variants of diffusion samplers. 
### Flow matching
Flow Matching (also known as rectified flow, or a special case of stochastic interpolant) provides another perspective of the forward process: Instead of viewing it as gradually adding noise to the clean data, we view it as an interpolation between the data $${\bf x}$$ and the Gaussian noise $$\boldsymbol \epsilon$$. In the more general case, it can be an interpolation of two arbitrary distributions. The forward process is further simplified as $${\bf z}_t = t {\bf x} + (1-t) {\boldsymbol \epsilon}$$. The evolvement of $${\bf z}_t$$ over time can be expressed as $${\bf z}_t = {\bf z}_{t - \Delta t} + ({\bf x} - {\boldsymbol \epsilon}) \Delta t$$, where $${\bf x} - {\bf \epsilon}$$ is the "velocity", "flow", or "vector field". For sampling, we simply do time reversal, and replace the vector field with our best guess at time step $$t$$ given $${\bf z}_t$$ (since we do not have access to $${\bf x}$$ during sampling):

$$
\begin{eqnarray}
{\bf z}_{t - \Delta t} = {\bf z}_t - (\hat{\bf x} - \hat{\boldsymbol \epsilon})\Delta t.\\
\end{eqnarray}
$$
$$\hat{\bf u} = \hat{\bf u}({\bf z}_t; t) := \hat{\bf x} - \hat{\boldsymbol \epsilon}$$ can be parametrized by a neural network.


So far we can already sense the similar flavors of the two frameworks:


<div style="background-color: lightyellow; padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. <strong>Same forward process</strong>: assume that one end of flow matching is Gaussian, and the noise schedule of diffusion models is in a particular form. </p>
  <p  style="margin: 0;">2. <strong>"Similar" sampling processes</strong>: both follow an iterative update that involves the unknown clean data, which is replaced by the best guess of the clean data at the current time step. (Spoiler: later we will show they are exactly the same!)</p>
</div>


## Training 

<!-- For training, a neural network is estimated to predict $$\hat{\boldsymbol \epsilon} = \hat{\boldsymbol \epsilon}({\bf z}_t; t)$$ that effectively estimates $${\mathbb E} [{\boldsymbol \epsilon} \vert {\bf z}_t]$$, the expected noise added to the data given the noisy sample. Other **model outputs** have been proposed in the literature which are linear combinations of $$\hat{\boldsymbol \epsilon}$$ and $${\bf z}_t$$, and $$\hat{\boldsymbol \epsilon}$$ can be derived from the model output given $${\bf z}_t$$.  -->

For diffusion models, <d-cite key="kingma2024understanding,hoogeboom2024simpler"></d-cite> summarize the training as estimating a neural network to predict $$\hat{\bf x} = \hat{\bf x}({\bf z}_t; t)$$, or a linear combination of $$\hat{\bf x}$$ and $${\bf z}_t$$. Learning the model is done by minimizing a weighted mean squared error (MSE) loss :
$$
\begin{equation}
\mathcal{L}(\mathbf{x}) = \mathbb{E}_{t \sim \mathcal{U}(0,1), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \textcolor{green}{w(\lambda_t)} \cdot \frac{\mathrm{d}\lambda}{\mathrm{d}t} \cdot \lVert\hat{\bf x} - {\bf x}\rVert_2^2 \right],
\end{equation}
$$
where $$\lambda_t$$ is the log signal-to-noise ratio, and $$\textcolor{green}{w(\lambda_t)}$$ is the **weighting function**, balancing the importance of the loss at different noise levels. The term $$\mathrm{d}\lambda / {\mathrm{d}t}$$ in the training objective seems unnatural and one may wonder why not merging it with the weighting function. As we will show below, this term helps *disentangle* the factors of noise schedule and weighting function clearly, and makes only one of them matter.  

To see why flow matching also fits in the above training objective, recall the conditional flow matching objective used by <d-cite key="lipman2022flow, liu2022flow"></d-cite> is

$$
\begin{equation}
\mathcal{L}_{\mathrm{CFM}}(\mathbf{x}) = \mathbb{E}_{t \sim \mathcal{U}(0,1), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \lVert \hat{\bf u} - {\bf u} \rVert_2^2 \right]
\end{equation}
$$

Since $$\hat{\bf u} = \hat{\bf x} - \hat{\boldsymbol{\epsilon}} = \hat{\bf x} - ({\bf z}_t - \alpha_t \hat{\bf x}) / \sigma_t$$ is a linear combination of $$\hat{\bf x}$$ and $${\bf z}_t$$, the CFM training objective can be rewritten as mean squared error on $${\bf x}$$ with a specific weighting. 

For training there are three design choices that are typically considered in the literature: training noise schedule, network output, and weighting function. There is often confusion in the field about how these choices affect the results and what the tuning recipe one should choose. We will elaborate them below.

### Network output
Below we summarize several network outputs proposed in the literature, including a few of diffusion models and the one of flow matching. One may see the training objective defined with different network outputs in different papers. From the perspective of training objective, they all correspond to having some additional weighting in front of the $${\bf x}$$-MSE that can be absorbed in the weighting function. 

| Network Output  | Formulation   | MSE on Network Output  |
| :------------- |:-------------:|-----:|
| $${\bf x}$$-prediction      | $$\hat{\bf x} $$      | $$ \lVert\hat{\bf x} - {\bf x}\rVert_2^2 $$ |
| $${\boldsymbol \epsilon}$$-prediction      |$$\hat{\boldsymbol \epsilon} = ({\bf z}_t - \alpha_t \hat{\bf x}) / \sigma_t$$ | $$\lVert\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\rVert_2^2 = e^{\lambda} \lVert\hat{\bf x} - {\bf x}\rVert_2^2 $$|
| $${\bf v}$$-prediction | $$\hat{\bf v} = \alpha_t \hat{\boldsymbol{\epsilon}} - \sigma_t \hat{\bf x} $$      |    $$ \lVert\hat{\bf v} - {\bf v}\rVert_2^2 = \sigma_t^2(e^{-\lambda} + 1)^2 \lVert\hat{\bf x} - {\bf x}\rVert_2^2 $$ |
| $${\bf u}$$-flow matching vector field | $$\hat{\bf u} = \hat{\bf x} - \hat{\boldsymbol{\epsilon}} $$      |    $$ \lVert\hat{\bf u} - {\bf u}\rVert_2^2 = (1 + e^{\lambda / 2})^2 \lVert\hat{\bf x} - {\bf x}\rVert_2^2 $$ |

In practice, however, the model output might make a difference. Specifically,
* $${\bf x}$$-prediction can be problematic at low noise levels. One reason is that the input sample at low noise levels is pretty close to the clean data and the optimization problem over $${\bf x}$$-MSE becomes almost trivial. The fine-grained details can be largely ignored by the model. The other reason is that any error in $$\hat{\bf x}$$ will get ampified in $$\hat{\boldsymbol \epsilon} = ({\bf z}_t - \alpha_t \hat{\bf x}) / \sigma_t$$, as $$\sigma_t$$ is close to 0.
* Following the similar reason, $${\boldsymbol \epsilon}$$-prediction can be problematic at high noise levels, i.e., the model prediction becomes not informative, and the error gets amplified in $$\hat{\bf x}$$.

Therefore, a heuristic is to choose a network output that is a combination of $${\bf x}$$- and $${\boldsymbol \epsilon}$$-prediction, which applies to the $${\bf v}$$-prediction and flow matching vector field. 


### Noise schedule
The noise schedule of flow matching is in a very simple form: $$\alpha_t = t, \sigma_t = 1 - t$$. Various noise schedules have been proposed in the diffusion literature, such as variance-preserving schedules ($$\alpha_t^2 + \sigma_t^2 = 1$$), variance-exploding schedules ($$\alpha_t=1$$), and other options in between. A few remarks about noise schedule:
1. All different noise schedules can be normalized as a variance-preserving schedule, with a linear scaling of $${\bf z}_t$$ and an unscaling at the network input. The key defining property of a noise schedule is the log signal-to-noise ratio $$\lambda_t$$.
2. The training loss is *invariant* to the training noise schedule, since the loss fuction can be rewritten as $$\mathcal{L}(\mathbf{x}) = \int_{\lambda_{\min}}^{\lambda_{\max}} w(\lambda) \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \|\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\|_2^2 \right] \, d\lambda$$, which is irrelevant to $$\lambda_t$$, and only related $$\lambda_{\mathrm{min}}$$ and $$\lambda_{\mathrm{max}}$$. However, $$\lambda_t$$ might still affect the variance of the Monte Carlo estimator of the training loss. A few heuristics have been proposed in the literature to automatically adjust the noise schedules over the training course. See [Sander's blog post](https://sander.ai/2024/06/14/noise-schedules.html#adaptive) for a nice summary.
3. As we will see in the next section, the testing noise schedule does impact the sample quality. However, one can choose completely different noise schedules for training and sampling, based on distinct heuristics: For training, it is desirable to have a noise schedule that minimizes the variance of the Monte Calor estimator, whereas for sampling the noise schedule is more related to the discretization error of the ODE / SDE sampling trajectories and the model curvature.


### Weighting function

Weighting function balances the importance of different noise levels during training, and effectively balances the importance of high frequency and low frequency components of the input signal. **(TODO, making a figure to illustrate weighting function versus frequency components.)** This is crucial for modeling perceptual signals such as images, videos and audios, as certain high frequency components in those signals are not visible to human perception, and thus better not to waste model capacity on them. We want to highlight one fact:

<div style="background-color: lightyellow; padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>For training objectives,</p>
  <p align="center" style="margin: 0;"><em>Flow matching == diffusion models with ${\bf v}$-MSE loss + cosine noise schedule.</em></p>
</div>


See Appendix D.2-3 in <d-cite key="kingma2024understanding"></d-cite> for a detailed derivation. Figure **TODO** plots several commonly used weighting functions in the literature. 

In summary, we have the following conclusions for diffusion models / flow matching training:

<div style="background-color: lightyellow; padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. Weighting function <strong>balances the importance of different frequency components in the data</strong>. Should tune based on the characteristics of the input data </p>
  <p>2. Noise schedule <strong>does not affect the training objective</strong> and only affects the training efficiency. As long as the endpoints are far enough it should not affect the results dramatically. Can use an adative noise schedule in the literature to speed up training. </p>
  <p style="margin: 0;">3. The <strong>network output proposed by flow matching is new</strong>. A network output that nicely balances ${\bf x}$- and ${\epsilon}$-prediction is desirable. </p>
</div>


## Sampling and Straightness Misnomer

<p align="center"><i>"Flow matching paths are straight, whereas diffusion paths are curved."</i></p>

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/interactive_alpha_sigma.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

<!-- given $${\bf z}_t$$, these following two updates are equivalent in distribution when $$s$$ is small: -->

### Deterministic sampler vs. stochastic sampler

So far we mainly cover the deterministic sampler of diffusion models or flow matching. An alternative is to use stochastic samplers such as the DDPM sampler <d-cite key="ho2020denoising"></d-cite>. The key is to realize that, the effect of a small step of DDIM update can be canceled out by a small step of forward diffusion update in distribution. To see why it is true, let's take a look at a 2D example. Starting from the same mixture of Gaussians distribution, we either apply a reverse DDIM update, or a diffusion update:
{% include figure.html path="assets/img/2025-04-28-distill-example/particle_movement.gif" class="img-fluid" %}
For each individual sample, the two updates are very different. The reverse DDIM update consistently drags every sample away from the modes of the distribution, while the diffusion update is purely random. However, aggregating all samples together, the distributions after the updates are the same. Therefore, running the DDIM update will get canceled out by the diffusion update. That means we can run DDIM update with a large step then followed by a renoising step, which matches the effect of running DDIM update with a smaller step. 

<!-- If we flip the sign of the drift update and add another diffusion update: $${\bf z}_{t+\Delta t} = {\bf z}_t + s \nabla_{\bf z} \log p_t({\bf z}) + \sqrt{2s}{\bf e}$$, the effect of the two updates gets canceled out, so that the distribution remains unchanged. The DDPM sampler or its variants essentially add certain amount of these two updates on top of the DDIM sampler at every time step. The benefit is that if the model prediction is not perfectly accurate, the diffusion update helps correct the error. -->
<!-- The formal proof requires some manipulation of Fokker-Planck equation <d-cite key="song2020score"></d-cite>.  -->
<!-- The drift update is about stretching or flattening the distribution, whereas the diffusion update is about smoothing or reducing the curvature of the distribution.   -->

<<<<<<< HEAD
In fact, performing one DDPM sampling step going from $\lambda_t$ to $\lambda_t + \Delta\lambda$ is equivalent to performing one DDIM sampling step to $\lambda_t + 2\Delta\lambda$, and then renoising to $\lambda_t + \Delta\lambda$ by doing forward diffusion (Emiel, do you want to write this out in detail somewhere?). DDPM thus reverses exactly half the progress made by DDIM, in terms of the log signal-to-noise ratio. However, the fraction of the DDIM step to undo by renoising is a hyperparameter which we are free to choose, and which has been called the level of _churn_ by [Insert citation to EDM paper]. The effect of adding churn to our sampler is to diminish the effect on our final sample of our model predictions made early during sampling, and to increase the weight on later predictions. This is shown in the Figure below

{% include figure.html path="assets/img/ddim_vs_ddpm.png" class="img-fluid" %}

Todo: replace this figure with an interactive version with a slider on the level of churn.

## From Diffusion Models to Flow Matching and back
=======
## From Diffusion Models to Flow Matching and back (WIP)
>>>>>>> 815fbae (modify sampling)

In this section, we show the equivalence between diffusion models and flow matching approaches from a stochastic process point of view. Note that it is possible to show this equivalence using other apporaches [CITE]


### Notation (to remove) 

$\mathbf{X}_t$
$\pi$
$\mathcal{N}(0, \mathrm{Id})$
my nitpick on the backward
$\mathbf{Y}_t$ for the backward process
$\alpha_t, \sigma_t, \varepsilon_t$ for FM
$f_t, g_t, \eta_t$ for DM
$A_t, S_t$ for the obtained interpolation from DM

We should pick a name (Flow Matching) and stick with it (but mention in the intro stochastici nterpolantetc)


### Flow Matching

In Flow Matching (and stochastic interpolant), we start by defining an interpolation. In order to be consistent with diffusion models notation, we will denote $\mathbf{X}_0 \sim \pi$, where $\pi$ is the data distribution and $\mathbf{Z}_1 \sim \mathcal{N}(0, \mathrm{Id})$. 

We start by defining the interpolation

$$
\mathbf{X}_t = \alpha_t \mathbf{X}_0 + \sigma_t \mathbf{Z} . 
$$

We would like to flow from $\mathbf{Z}$ to $\mathbf{X}_0$.
The associated ODE is given by 

$$
\mathrm{d} \mathbf{X}_t = \left{ \dot{\alpha}_t \mathbf{X}_0 + \dot{\sigma}_t \mathbf{Z} \right} \mathrm{d} t .
$$

Since, we want to flow from the noise to the data, we define $\mathbf{Y}_t = \mathbf{X}_{1-t}$ and we get that 

$$
\mathrm{d} \mathbf{Y}_t = - \left{ \dot{\alpha}_{1-t} \mathbf{Y}_1 + \dot{\sigma}_{1-t} \mathbf{Y}_0 \right} \mathrm{d} t .
$$

Of course, at inference time we do not have access to $\mathbf{Y}_1$, i.e., we do not have access to the original datapoint $\mathbf{X}_0$. Instead, we replace it by our best guess at time $t$. This is equivalent to consider the conditional expectation of the velocity given $\mathbf{Y}_t$. Hence, we define 

$$
\mathrm{d} \mathbf{Y}_t = - \{ \mathbb{E}[\dot{\alpha}_{1-t} \mathbf{Y}_1 + \dot{\sigma}_{1-t} \mathbf{Y}_0 | \mathbf{Y}_t] \} \mathrm{d} t  . 
$$

Give theorem to say why this is true?

Usually, the quantity $\mathbb{E}[\dot{\alpha}_{t} \mathbf{X}_0 + \dot{\sigma}_{t} \mathbf{X}_1 | \mathbf{X}_t = x] = v_t(x)$ is called the velocity flow matching and can be learned via regression loss. It corresponds to the loss ... 

There is exists a one-to-one mapping between the flow matching velocity and the score function given by 

$$
v_t(x) = \tfrac{\dot{\alpha}_t}{\alpha_t} x - \sigma_t ( \dot{\sigma}_t - \tfrac{\dot{\alpha}_t}{\alpha_t} \sigma_t) \nabla \log p_t(x)
$$

Using the Fokker-Planck trick (maybe have it in the appendix, add Amsterdam (TM) remark that this corresponds to say that stochastic = deterministic + renoising) we can get a stochastic generative process

$$
\mathrm{d} \mathbf{Y}_t = \{ -v_{1-t}(\mathbf{Y}_t) +\tfrac{\varepsilon_{1-t}^2}{2} \nabla \log p_{1-t}(\mahtbf{Y}_t) \} \mathrm{d} t + \varepsilon_{1-t}  \mathrm{d} \mathbf{B}_t ,
$$

where $(\mathbf{B}_t)_{t \in [0,1]}$ is a $d$-dimensional Brownian motion (maybe here give a natural interpretation of the Brownian motion and SDE, using the discretisation, talk about exponential integrators?)

Hence in flow matching we have three free parameters:
* $\alpha_t$ -- smol description
* $\sigma_t$ -- smol description
* $\varepsilon_t$ -- smol description


(ruiqi backup)

In the more general case, it can be an interpolation of two arbitrary distributions. We can therefore get an associated ODE:

$$
\begin{eqnarray}
\mathrm{d} \mathbf{z}_t = \left[ \dot{\alpha}_t \mathbf{x} + \dot{\sigma}_t {\boldsymbol \epsilon} \right] \mathrm{d} t = {\bf u_t} \mathrm{d} t,
\end{eqnarray}
$$
where $${\bf u}_t$$ is called the *flow matching vector field* at time t. Inverting this ODE for generating samples seems to be trivial, just run the ODE backward in time. However, at inference we do not have access to $${\bf x}$$, i.e., the datapoint, neither to the ground truth vector field $${\bf u}_t$$. We can use a vector field network $$\hat{\bf u} = \hat{\bf u}({\bf z}_t; t)$$ to esimate that, and again we need to define a valid target for that (the ground truth marginal vector field is unknown). Flow matching proves that, assume $${\bf u} ({\bf z}_t \lvert {\bf x})$$ is a conditional vector field  that generates the conditional probablity path $$p_t(\cdot \lvert {\bf x})$$, then $$\mathbb{E}_{p({\bf x}\lvert{\bf z}_t)} \left[ {\bf u}({\bf z}_t \lvert {\bf x}) \right]$$ generates the marginal probability path $$p_t(\cdot)$$. As a result, we can use the conditional vector field $${\bf u} ({\bf z}_t \lvert {\bf x}) = \dot{\alpha}_t \mathbf{x} + \dot{\sigma}_t {\boldsymbol \epsilon}$$ as the target.


### Diffusion models


In diffusion models we usually define a forward process that we try to reverse.

$$
\mathrm{d} \mathbf{X}_t = f_t \mathbf{X}_t \mathrm{d} t + g_t \mathrm{d} \mathbf{B}_t .
$$

Associated with this SDE is the following integral representation

$$
\mathbf{X}_t = A_t \mathbf{X}_0 + S_t Z , \qquad Z \sim \mathrm{N}(0, \mathrm{Id}) . 
$$

where $A_t$ and $S_t$ are explicit functions of $f_t$ and $g_t$ and can be obtained as

$$
A_t = \exp[\int_0^t f_s \mathrm{d}s] , \qquad S_t = \int_0^t g_s^2 \exp[-2\int_0^s f_u \mathrm{d}u] \mathrm{d} s . 
$$

The backward model associated with the forward SDE is given by 

$$
\mathrm{d} Y_t = \{ -f_{1-t} \mathbf{Y}_t + g_{1-t}^2 \nabla \log p_{1-t}(\mathbf{Y}_t) \} \mathrm{d} t + g_{1-t} \mathrm{d} \mathbf{B}_t .
$$

Using Tweedie's formula we get that 

$$
\nabla \log p_t(x) = \mathbb{E}[(A_t \mathbf{X}_0 - \mathbf{X}_t) / S_t | \mathbf{X}_t = x] .
$$

Similarly to the flow matching setting, we can play with the level of stochasticity.
We introduce an additional parameter $\eta_t$ and we get a modified version of ... 

$$
\mathrm{d} Y_t = \{ -f_{1-t} \mathbf{Y}_t + (1 + \eta_{1-t})\tfrac{g_{1-t}^2}{2} \nabla \log p_{1-t}(\mathbf{Y}_t) \} \mathrm{d} t + \eta_{1-t} g_{1-t} \mathrm{d} \mathbf{B}_t .
$$

Hence in flow matching we have three free parameters:
* $f_t$ -- smol description
* $g_t$ -- smol description
* $\eta_t$ -- smol description


(ruiqi backup)

The diffusion process can also be expressed as a forward stochastic differential equation (SDE): 

$$
\begin{equation}
\mathrm{d}{\bf z} = f(t) {\bf z} \mathrm{d}t + g(t) \mathrm{d}{\bf w},
\end{equation}
$$
where $${\bf w}$$ is a standard Wiener process, and 
$$
\begin{equation}
f(t) = \frac{\mathrm{d} \log \alpha_t}{\mathrm{d}t}, \; g^2(t) = \frac{\mathrm{d} \sigma_t^2}{\mathrm{d}t} - 2 \frac{\mathrm{d} \log \alpha_t}{\mathrm{d}t}\sigma_t^2
\end{equation}
$$
For generating samples, one can solve a SDE that runs backwards in time and exactly revserses the forward SDE:
$$
\begin{equation}
d{\bf z} = \left[f(t) {\bf z} - g(t)^2 \nabla _{\bf z} \log p_t({\bf z}) \right] dt + g(t) \mathrm{d}{\bf w},
\end{equation}
$$
where $$\nabla _{\bf z} \log p_t({\bf z})$$ is the *score function* at time $$t$$.
An alternative is to solve a probability flow ODE that gives the same marginal distribution as the reverse SDE at every time step $$t$$:
$$
\begin{equation}
\mathrm{d}{\bf z} = \left[f(t) {\bf z} - \frac{1}{2} g(t)^2 \nabla _{\bf z} \log p_t({\bf z}) \right] \mathrm{d}t.
\end{equation}
$$
Both sampling equations involve estimating the score function that we cana parametrize as a score network $$\hat{\bf s} = \hat{\bf s}({\bf z}_t; t)$$, but we don't have access to the ground truth score to use as the target during training. Thankfully, we observe that $$\nabla _{\bf z} \log p_t({\bf z}) = \mathbb{E}_{p({\bf x} | {\bf z}_t)}\left[  \log p({\bf z}_t | {\bf x}) \right]$$. Therefore, we can instead use the conditioinal score $$\nabla _{\bf z}\log p({\bf z}_t | {\bf x}) = - {\boldsymbol \epsilon} / \sigma_t$$ as the target of the score network, and one can show that minimizing a mean squared error loss with the conditional score is equivalent to miminizing the same loss with the marginal score in expectation.


### How to relate these two models?

As, we have seen flow matching depends on three parameters $\alpha_t$, $\sigma_t$, $\varepsilon_t$ and diffusion models depend on three parameters as well $f_t$, $g_t$, $\eta_t$. In this section, we are going to show that one can define a one-to-one mapping between those quantities so that flow matching and diffusion models define exactly the same model. 

First, we notice that diffusion models define an interpolant using ...

Hence, if we want to make the two models identical we need to set $A_t = \alpha_t$ and $S_t = \sigma_t$. 
This will give us a relation between $f_t, g_t$ and $\alpha_t, \sigma_t$. 

We get the following relationship

$$
f_t = \partial_t \log(\alpha_t) , \qquad g_t = (2 \alpha_t \sigma_t \partial_t (\sigma_t / \alpha_t))^{1/2} .
$$

At this stage, we have found a way to define a forward SDE in a diffusion model framework that yields an interpolant defined by flow matching using ... Similarly, using ... we can define an interpolant in the flow matching framework given a forward SDE given by a diffusion model.

So far, we have shown that by defining a relationship between $f_t, g_t$ and $\alpha_t, \sigma_t$ we can define the same interpolant and forward noising process in both the flow matching and the diffusion model frameworks. 

However, one question remains, do the generative SDEs coincides? Recall that the general generative SDE is given by ... in the flow matching framework and by ... in the diffusion model framework. 

Let's have a look at the coefficients in front of the Brownian motion in both cases. In the flow matching setting, we have $\varepsilon_{1-t}$ and in the diffusion model setting we have $\eta_{1-t} g_{1-t}$. This suggests to set 

$$
\varepsilon_t = \eta_t g_t . 
$$

Sure, this makes the coefficients in front of the  Brownian motion equal but what about the drift terms?
Well, it turns out that with the relationships ... and ... they are also equal and therefore the dynamics are identical!

This is the content of the following theorem

...

To summarize, given $\alpha_t$, $\sigma_t$ and $\varepsilon_t$ defining a flow matching framework, we can get an equivalent diffusion model framework by defining 

$$
f_t = \partial_t \log(\alpha_t) , \qquad g_t = 2 \alpha_t sgima_t \partial_t (\sigma_t / \alpha_t) , \qquad \eta_t = \varepsilon_t / [2 \alpha_t sgima_t \partial_t (\sigma_t / \alpha_t)]
$$

Similarly, given $f_t$, $g_t$ and $\eta_t$ defining a diffusion model framework, we can get an equivalent flow matching framework by defining 

$$
\alpha_t = \exp[\int_0^t f_s \mathrm{d}s] , \qquad \sigma_t = (\int_0^t g_s^2 \exp[-2\int_0^s f_u \mathrm{d}u] \mathrm{d} s)^{1/2} , \qquad \varepsilon_t = \eta_t g_t . 
$$

