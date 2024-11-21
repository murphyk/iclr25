---
layout: distill
title: "Diffusion Models and Gaussian Flow Matching: Two sides of the same coin."
description: "Flow matching and diffusion models are two popular frameworks in generative modeling. Despite seeming similar, there is general confusion in the community about their exact connection. In this post we aim to clear up this confusion and show that <i>diffusion model and Gaussian flow matching are essentially the same</i>: Different model specifications have different noise schedules and loss weighting but correspond to the same generative model. That's great news, it means that you can use the two frameworks interchangeably."
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


Flow matching is gaining popularity, due to its simplicity in formulation and "straightness" in the sampling trajectories. A common question one hears nowadays is: 


<!-- > Does this diffusion technique also work with Gaussian flow matching? -->
<p align="center"><i>"Does this diffusion technique also work with flow matching?"</i></p>


What exactly are the differences between these two approaches? As we will see, diffusion modelling and Gaussian flow matching are nearly the same. So the answer to this question is "yes", unless the matching is not to a Gaussian.

To give an example, you may assume that flow matching sampling has to be deterministic. However, you have trained a general denoiser: stochastic or deterministic sampling, it's up to you!


In this blog post, we take the most commonly<d-footnote>We focus on Gaussian flow matching with the optimal transport flow path.</d-footnote> used flow matching case <d-cite key="lipman2022flow"></d-cite>, also very related to <d-cite key="liu2022flow"></d-cite> and <d-cite key="albergo2023stochastic"></d-cite>. Our purpose is not to downweigh the importance of either framework. In fact, both frameworks are important and are derived from distinct theoretical perspectives. It is even more encouraging that they lead to the same algorithm in practice. The goal of this post is to make the practitioner feel comfortable to use the two frameworks interchangeably, understand the actual degrees of freedom we have when tuning the algorithm (no matter how we name it).


## Overview

We start by recalling the two frameworks (diffusion models and flow matching). We compare them from a high level and will see that the *process* is the same.
<!-- We highlight the free parameters in each framework and how they relate to each other. 
In particular, there exists explicit mappings to define a diffusion model from a flow matching model and vice-versa. This overview does not dive into the training of such models, i.e., we assume that all the learnable quantities have been adequatly optimized. We also do not discuss the different sampling techniques used at inference. Both the training and the inference will be discussed in further sections.  -->


### Diffusion models

A diffusion process gradually destroys an observed data $$ \bf{x} $$ over time $$t$$, by mixing the data with Gaussian noise. Summing up this noise over time gives:
$$
\begin{equation}
{\bf z}_t = \alpha_t {\bf x} + \sigma_t {\boldsymbol \epsilon}, \;\mathrm{where} \; {\boldsymbol \epsilon} \sim \mathcal{N}(0, {\bf I}).
\label{eq:forward}
\end{equation}
$$
$$\alpha_t$$ and $$\sigma_t$$ define the **noise schedule**. A useful notation is the log signal-to-noise ratio $$\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$$, which decreases as $$t$$ increases from $$0$$ (clean data) to $$1$$ (Gaussian noise).

To generate new samples, we can "reverse" the forward process gradually: Initialize the sample from Gaussian at the highest noise level. Given the sample $${\bf z}_t$$ at time step $$t$$, we predict what the clean sample might look like with a neural network $$\hat{\bf x} = \hat{\bf x}({\bf z}_t; t)$$, and then we project it back to a lower noise level with the same forward transformation:

$$
\begin{eqnarray}
{\bf z}_{t - \Delta t} &=& \alpha_{t - \Delta t} \hat{\bf x} + \sigma_{t - \Delta t} \hat{\boldsymbol \epsilon},\\
\end{eqnarray}
$$
where $$\hat{\boldsymbol \epsilon} = ({\bf z}_t - \alpha_t \hat{\bf x}) / \sigma_t$$. We keep alternating between predicting the clean data, and projecting it back to a lower noise level until we get the clean sample.
This is the DDIM sampler <d-cite key="song2020denoising"></d-cite>. The randomness of samples only comes from the initial Gaussian sample, and the entire reverse process is deterministic. 

### Flow matching
Flow Matching provides another perspective of the forward process: we view it directly as an interpolation between the data $${\bf x}$$ and the Gaussian noise $$\boldsymbol \epsilon$$. In the more general case, $$\boldsymbol \epsilon$$ can also be sampled from an arbitrary distribution. The forward process should look familiar to the reader, and is defined as:
$$
\begin{eqnarray}
{\bf z}_t = t {\bf x} + (1-t) {\boldsymbol \epsilon}.\\
\end{eqnarray}
$$

The flow of $${\bf z}_t$$ can be expressed as $${\bf z}_t = {\bf z}_{t - \Delta t} + ({\bf x} - {\boldsymbol \epsilon}) \Delta t$$, where $${\bf x} - {\bf \epsilon}$$ is the "velocity", "flow", or "vector field". For sampling, we reverse terms and replace the vector field with our best guess $$\hat{\bf x}$$ at time step $$t$$ given $${\bf z}_t$$ (since we do not have access to $${\bf x}$$ during sampling):

$$
\begin{eqnarray}
{\bf z}_{t - \Delta t} = {\bf z}_t - (\hat{\bf x} - \hat{\boldsymbol \epsilon})\Delta t.\\
\end{eqnarray}
$$
$$\hat{\bf u} = \hat{\bf u}({\bf z}_t; t) := \hat{\bf x} - \hat{\boldsymbol \epsilon}$$ can be parametrized by a neural network.


So far we can already sense the similar flavors of the two frameworks:


<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. <strong>Same forward process</strong>: assume that one end of flow matching is Gaussian, and the noise schedule of diffusion models is in a particular form. </p>
  <p  style="margin: 0;">2. <strong>"Similar" sampling processes</strong>: both follow an iterative update that a guess of the clean data at the current time step. (Spoiler: later we will show they are exactly the same!)</p>
</div>


If the process is the same, what about the training?

## Training (weighting, output, schedule)

<!-- For training, a neural network is estimated to predict $$\hat{\boldsymbol \epsilon} = \hat{\boldsymbol \epsilon}({\bf z}_t; t)$$ that effectively estimates $${\mathbb E} [{\boldsymbol \epsilon} \vert {\bf z}_t]$$, the expected noise added to the data given the noisy sample. Other **model outputs** have been proposed in the literature which are linear combinations of $$\hat{\boldsymbol \epsilon}$$ and $${\bf z}_t$$, and $$\hat{\boldsymbol \epsilon}$$ can be derived from the model output given $${\bf z}_t$$.  -->

Diffusion models <d-cite key="kingma2024understanding"></d-cite> are trained by estimating $$\hat{\bf x} = \hat{\bf x}({\bf z}_t; t)$$ with a neural net. In practice, one chooses a linear combination of $$\hat{\bf x}$$ and $${\bf z}_t$$ for stability reasons.
<!-- <d-footnote>It take a little bit of effort to show that indeed you only need linear combinations to define model outputs such as $$\hat{\boldsymbol{\epsilon}}$$, $$\hat{\bf v}$$ and $$\hat{\bf u}$$ (from flow matching)</d-footnote>. -->
Learning the model is done by minimizing a weighted mean squared error (MSE) loss:
$$
\begin{equation}
\mathcal{L}(\mathbf{x}) = \mathbb{E}_{t \sim \mathcal{U}(0,1), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \textcolor{green}{w(\lambda_t)} \cdot \frac{\mathrm{d}\lambda}{\mathrm{d}t} \cdot \lVert\hat{\bf x} - {\bf x}\rVert_2^2 \right],
\end{equation}
$$
where $$\lambda_t$$ is the log signal-to-noise ratio, and $$\textcolor{green}{w(\lambda_t)}$$ is the **weighting function**, balancing the importance of the loss at different noise levels. The term $$\mathrm{d}\lambda / {\mathrm{d}t}$$ in the training objective seems unnatural and in literature is often merged with the weighting function. However, their separation helps *disentangle* the factors of noise schedule and weighting function clearly, and helps emphasize the more important weighting components.  

Flow matching also fits in the this training objective, recall the conditional flow matching objective used by <d-cite key="lipman2022flow, liu2022flow"></d-cite> is

$$
\begin{equation}
\mathcal{L}_{\mathrm{CFM}}(\mathbf{x}) = \mathbb{E}_{t \sim \mathcal{U}(0,1), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \lVert \hat{\bf u} - {\bf u} \rVert_2^2 \right]
\end{equation}
$$

Since $$\hat{\bf u} = \hat{\bf x} - \hat{\boldsymbol{\epsilon}} = \hat{\bf x} - ({\bf z}_t - t \hat{\bf x}) / (1-t)$$ is a linear combination of $$\hat{\bf x}$$ and $${\bf z}_t$$, the CFM training objective can be rewritten as mean squared error on $${\bf x}$$ with a specific weighting. 


### What's the weight?
The weighting is the most important part of the loss, it balances the importance of high frequency and low frequency components.  **(TODO, making a figure to illustrate weighting function versus frequency components.)** 
This is important when modeling images, videos and audios, as certain high frequency components in those signals are not visible to human perception, and thus better not to waste model capacity on them. Viewing losses via their weighting, one can derive that for example:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <!-- <p>For weighting functions,</p> -->
  <p align="center" style="margin: 0;"><em>Flow matching weighting == diffusion weighting of ${\bf v}$-MSE loss + cosine noise schedule.</em></p>
</div>

See Appendix D.2-3 in <d-cite key="kingma2024understanding"></d-cite> for a detailed derivation. Figure **TODO** plots several commonly used weighting functions in the literature. 

### Network output
Below we summarize several network outputs proposed in the literature, including a few of diffusion models and the one of flow matching. One may see the training objective defined with different network outputs in different papers. From the perspective of training objective, they all correspond to having some additional weighting in front of the $${\bf x}$$-MSE that can be absorbed in the weighting function. 

| Network Output  | Formulation   | MSE on Network Output  |
| :------------- |:-------------:|-----:|
| $${\bf x}$$-prediction      | $$\hat{\bf x} $$      | $$ \lVert\hat{\bf x} - {\bf x}\rVert_2^2 $$ |
| $${\boldsymbol \epsilon}$$-prediction      |$$\hat{\boldsymbol \epsilon} = ({\bf z}_t - \alpha_t \hat{\bf x}) / \sigma_t$$ | $$\lVert\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\rVert_2^2 = e^{\lambda} \lVert\hat{\bf x} - {\bf x}\rVert_2^2 $$|
| $${\bf v}$$-prediction | $$\hat{\bf v} = \alpha_t \hat{\boldsymbol{\epsilon}} - \sigma_t \hat{\bf x} $$      |    $$ \lVert\hat{\bf v} - {\bf v}\rVert_2^2 = \sigma_t^2(e^{-\lambda} + 1)^2 \lVert\hat{\bf x} - {\bf x}\rVert_2^2 $$ |
| $${\bf u}$$-flow matching vector field | $$\hat{\bf u} = \hat{\bf x} - \hat{\boldsymbol{\epsilon}} $$      |    $$ \lVert\hat{\bf u} - {\bf u}\rVert_2^2 = (1 + e^{\lambda / 2})^2 \lVert\hat{\bf x} - {\bf x}\rVert_2^2 $$ |

In practice, however, the model output might make a difference. For example,
* $${\bf x}$$-prediction can be problematic at low noise levels, because small changes create a large loss under typical weightings. You can also see in the sampler that any error in $$\hat{\bf x}$$ will get ampified in $$\hat{\boldsymbol \epsilon} = ({\bf z}_t - \alpha_t \hat{\bf x}) / \sigma_t$$, as $$\sigma_t$$ is close to 0.
* Following the similar reason, $${\boldsymbol \epsilon}$$-prediction is problematic at high noise levels, because $$\hat{\boldsymbol \epsilon}$$ is not informative, and the error gets amplified in $$\hat{\bf x}$$.

Therefore, a heuristic is to choose a network output that is a combination of $${\bf x}$$- and $${\boldsymbol \epsilon}$$-prediction, which applies to the $${\bf v}$$-prediction and the flow matching vector field $${\bf u} = {\bf x} - {\bf \epsilon}$$.


In summary, we have the following conclusions for diffusion models / flow matching training:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. Weighting function <strong> is important for training</strong>. For perceptual signals, it balances the importance of different frequency components. Should tune based on the characteristics of the input data. </p>
  <p>2. Noise schedule <strong>is far less important training objective</strong> and affects the training efficiency.</p>
  <p style="margin: 0;">3. The network output proposed by flow matching nicely balances ${\bf x}$- and ${\epsilon}$-prediction, similar to ${\bf v}$-prediction.</p>
</div>


### Noise schedule
The noise schedule of flow matching is in a very simple form: $$\alpha_t = t, \sigma_t = 1 - t$$. Various noise schedules have been proposed in the diffusion literature, such as variance-preserving schedules ($$\alpha_t^2 + \sigma_t^2 = 1$$), variance-exploding schedules ($$\alpha_t=1$$), and other options in between. A few remarks about noise schedule:
1. All different noise schedules can be normalized as a variance-preserving schedule, with a linear scaling of $${\bf z}_t$$ and an unscaling at the network input. The key defining property of a noise schedule is the log signal-to-noise ratio $$\lambda_t$$.
2. The training loss is *invariant* to the training noise schedule, since the loss fuction can be rewritten as $$\mathcal{L}(\mathbf{x}) = \int_{\lambda_{\min}}^{\lambda_{\max}} w(\lambda) \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \|\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\|_2^2 \right] \, d\lambda$$, which is irrelevant to $$\lambda_t$$, and only related $$\lambda_{\mathrm{min}}$$ and $$\lambda_{\mathrm{max}}$$. However, $$\lambda_t$$ might still affect the variance of the Monte Carlo estimator of the training loss. A few heuristics have been proposed in the literature to automatically adjust the noise schedules over the training course. See [Sander's blog post](https://sander.ai/2024/06/14/noise-schedules.html#adaptive) for a nice summary.
3. As we will see in the next section, the testing noise schedule does impact the sample quality. However, one can choose completely different noise schedules for training and sampling, based on distinct heuristics: For training, it is desirable to have a noise schedule that minimizes the variance of the Monte Calor estimator, whereas for sampling the noise schedule is more related to the discretization error of the ODE / SDE sampling trajectories and the model curvature.



## Sampling and Straightness Misnomer

<!-- <p align="center"><i>"Flow matching paths are straight, whereas diffusion paths are curved."</i></p>

Sampling from a score model is affected by a number of design choices.
Let's focus for now on determinsitic sampling where we want to use our trained score model to transform random noise into a datapoint.

In both frameworks deterministic sampling comes down to integrating an ODE. This ODE however is not unique because we must choose how to interplate between data and noise.
Once we have an ODE we must also pick a numerical method to compute it. The DDIM method analtyically integrates the sampling ODE for a constant prediction from your score method.

$$
DDIM(z_s | z_t) = \alpha_s * \hat{x} + \sigma_s * \hat{\epsilon}
$$

Others have used the standard Euler method or higher-order methods.
DDIM has the nice property that the resulting sample is the same whatever choice we make for interpolating data and noise.
This is generally not the case for euler or higher order integrators.
However, for the standard flow matching interpolation ($\alpha_t = 1. - t$, $\sigma_t = t$) Euler integration gives exactly the same results as DDIM.

So why is the flow matching paramterization said to result in straighter sampling paths?
When the model is perfectly confident about the data point it is moving towards, the path from noise to data will be a straight line.
Straight lined ODEs are great because it means that there is no integration error whatsover.
Unfortanely, typical score models are not modelling a single point. Instead they predict the average over a larger distribution.
In this case, there is no garantuee that the flow matching formulation or DDIM integration leads to less error.
In fact, in the interactive graph below we can see that the variance preserving formulation is optimal if the model prediction has a variance of $1$.

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/interactive_vp_vs_flow.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>


Finding such straight paths for real-life datasets like images is of course much less straightforward. But the conclusion remains the same: The optimal integration method depends on the data and the models prediction.

In the graph below you can tune the integration paths yourself.

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/interactive_alpha_sigma.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

Note also how the paths for DDIM will bend but the final datapoint it ends up predicting remain the same. -->

Thus, we can conclude a few things from determinstic sampling:
<!-- 1. For DDIM the interpolation between data and noise is irrelevant and always equivalant to flow matching <d-footnote>The variance exploding formulation ($\alpha_t = 1$, $\sigma_t = t$) is also equivalant to DDIM and flow matching.</d-footnote>. -->
2. Flow matching is only straight for a model predicting a single point. For realistic distributions other interpolations can give straighter paths.


<!-- given $${\bf z}_t$$, these following two updates are equivalent in distribution when $$s$$ is small: -->



### Reflow operator

The Reflow operation in Flow Matching connects noise and data points to sample in a straight line.
One can obtain these data noise pairs by running a deterministic sampler from noise.
A model can then be trained to directly predict the data given the noise avoiding the need for sampling.
In the diffusion literature the same approach was the one of the first distillation techniques <d-cite key="luhman2021knowledge"></d-cite>.




### Deterministic sampler vs. stochastic sampler

So far we mainly cover the deterministic sampler of diffusion models or flow matching. An alternative is to use stochastic samplers such as the DDPM sampler <d-cite key="ho2020denoising"></d-cite>. The key is to realize that, the effect of a small step of DDIM update can be canceled out by a small step of forward diffusion update in distribution. To see why it is true, let's take a look at a 2D example. Starting from the same mixture of Gaussians distribution, we either apply a reverse DDIM update, or a diffusion update:
{% include figure.html path="assets/img/2025-04-28-distill-example/particle_movement.gif" class="img-fluid" %}
For each individual sample, the two updates are very different. The reverse DDIM update consistently drags every sample away from the modes of the distribution, while the diffusion update is purely random. However, aggregating all samples together, the distributions after the updates are the same. Therefore, running the DDIM update will get canceled out by the diffusion update. That means we can run DDIM update with a large step then followed by a renoising step, which matches the effect of running DDIM update with a smaller step. 

<!-- If we flip the sign of the drift update and add another diffusion update: $${\bf z}_{t+\Delta t} = {\bf z}_t + s \nabla_{\bf z} \log p_t({\bf z}) + \sqrt{2s}{\bf e}$$, the effect of the two updates gets canceled out, so that the distribution remains unchanged. The DDPM sampler or its variants essentially add certain amount of these two updates on top of the DDIM sampler at every time step. The benefit is that if the model prediction is not perfectly accurate, the diffusion update helps correct the error. -->
<!-- The formal proof requires some manipulation of Fokker-Planck equation <d-cite key="song2020score"></d-cite>.  -->
<!-- The drift update is about stretching or flattening the distribution, whereas the diffusion update is about smoothing or reducing the curvature of the distribution.   -->

In fact, performing one DDPM sampling step going from $\lambda_t$ to $\lambda_t + \Delta\lambda$ is exactly equivalent to performing one DDIM sampling step to $\lambda_t + 2\Delta\lambda$, and then renoising to $\lambda_t + \Delta\lambda$ by doing forward diffusion. DDPM thus reverses exactly half the progress made by DDIM in terms of the log signal-to-noise ratio. However, the fraction of the DDIM step to undo by renoising is a hyperparameter which we are free to choose, and which has been called the level of _churn_ by <d-cite key="karras2022elucidating"></d-cite>. The effect of adding churn to our sampler is to diminish the effect on our final sample of our model predictions made early during sampling, and to increase the weight on later predictions. This is shown in the Figure below

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/churn.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

Here we ran different samplers for 100 sampling steps using a cosine noise schedule and v-prediction <d-cite key="salimansprogressive"></d-cite>. Ignoring nonlinear interactions, the final sample produced by the sampler can be written as a weighted sum of predictions made during sampling and noise. The weights of these predictions are shown on the y-axis for different diffusion times shown on the x-axis. DDIM results in an equal weighting of v-predictions for this setting, as shown by Salimans & Ho, whereas DDPM puts more emphasis on predictions made towards the end of sampling. Also see <d-cite key="lu2022dpm"></d-cite> for analytic expressions of these weights in the x and $\epsilon$ parameterizations.

## From Diffusion Models to Flow Matching and back

Finally, we provide useful formula to move from a Flow Matching point of view to a diffusion model point of view. 
We do not discuss training and sampling issues here but solely focus on showing that both frameworks are interchangeable. 

### Diffusion Models framework hyperparameters

Recalling CITE OVERVIEW section, we have that a diffusion model is defined by a forward process of the form 

$$
\begin{equation}
\mathrm{d} {\bf z}_t = f_t {\bf z}_t \mathrm{d} t + g_t \mathrm{d} {\bf z} .
\end{equation}
$$

Hence, the free parameters are given by $f_t$ and $g_t$. From the diffusion model perspective, the generative process is given by the backward of the forward process, i.e.

$$
\begin{equation}
\mathrm{d} {\bf z}_t = (f_t {\bf z}_t - \frac{1+ \eta_t^2}{2}g_t^2 \nabla \log p_t({\bf z_t}) ) \mathrm{d} t + \eta_t g_t \mathrm{d} {\bf z} .
\end{equation}
$$

Note that we have introduced an additional parameter $\eta_t$ which controls the amount of stochasticity at inference time. When discretizing the backward process we recover DDIM in the case $\eta_t = 0$ and DDPM in the case $\eta_t = 1$.

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  Diffusion model frameworks are entirely determined by three hyperparameters  
  <p>1. $f_t$ which controls how much we forget the original data in the forward process. </p>
  <p>2. $g_t$ which controls how much noise we input into the samples in the forward process. </p>
  <p style="margin: 0;">3. $\eta_t$ which controls the amount of stochasticity at inference time. </p>
</div>

### Flow Matching framework hyperparameters

Recalling CITE OVERVIEW section, we have that a diffusion model is defined by a forward process of the form 

$$
\begin{equation}
{\bf z}_t = \alpha_t {\bf x} + \sigma_t {\bf z} .
\end{equation}
$$

Hence, the free parameters are given by $\alpha_t$ and $\sigma_t$. From the flow matching perspective, the generative process is by the following trajectory

$$
\begin{equation}
\mathrm{d} {\bf z}_t = (v_t({\bf z_t}) - \varepsilon_t^2 \nabla \log p_t({\bf z_t})) \mathrm{d} t + \varepsilon_t \mathrm{d} {\bf z} .
\end{equation}
$$

Note that we have introduced an additional parameter $\varepsilon_t$ which controls the amount of stochasticity at inference time. 

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  Flow matching frameworks are entirely determined by three hyperparameters  
  <p>1. $\alpha_t$ which controls the data component in the interpolation. </p>
  <p>2. $\sigma_t$ which controls the noise component in the interpolation. </p>
  <p style="margin: 0;">3. $\varepsilon_t$ which controls the amount of stochasticity at inference time. </p>
</div>

### Equivalence of the points of view

Despite their clear similarities it is not immediately clear how to link the diffusion model framework and the flow matching one. 
Below, we provide formula which provide a one-to-one mapping between the two frameworks. In short:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  Diffusion model and flow matching are just one change of variable away!
</div>

Given a diffusion model framework, i.e. hyperparameters $f_t, g_t, \eta_t$ one can define 

$$
\begin{equation}
\alpha_t = \exp(\int_0^t f_s \mathrm{d}s) , \qquad \sigma_t = (\int_0^t g_s^2 \exp(-2\int_0^s f_u \mathrm{d}u) \mathrm{d} s)^{1/2} , \qquad \varepsilon_t = \eta_t g_t . 
\end{equation}
$$

Doing so, the noising process induced by the flow matching and the diffusion framework as well as the generative trajectories!
Similarly, given a flow matching framework, i.e. hyperparameters $\alpha_t, \sigma_t, \varepsilon_t$ one can define 

$$
f_t = \partial_t \log(\alpha_t) , \qquad g_t = 2 \alpha_t \sigma_t \partial_t (\sigma_t / \alpha_t) , \qquad \eta_t = \varepsilon_t / (2 \alpha_t \sigma_t \partial_t (\sigma_t / \alpha_t)) . 
$$

We have the similar conclusion, that under this transformation, diffusion models and flow matching frameworks coincide. 
To sum up, leaving aside training issues and the choice of the sampler, there is no fundamental differences between the two approaches.
