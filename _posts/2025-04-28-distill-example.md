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
  - name: Equations
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

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


## Training 

We will start by introducing a general training framework, and then discuss how flow matching and diffusion models fit in this framework. A diffusion process gradually destroys an observed data $$ \bf{x} $$ over time $$t$$, by mixing the data with Gaussian noise:

$$
{\bf z}_t = \alpha_t {\bf x} + \sigma_t {\boldsymbol \epsilon}, \;\mathrm{where} \; {\boldsymbol \epsilon} \sim \mathcal{N}(0, {\bf I}).
$$

$$\alpha_t$$ and $$\sigma_t$$ define the **noise schedule**. A useful notation is the log signal-to-noise ratio $$\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$$, which monotonically decreases as $$t$$ increases (i.e., goes from clean data to Gaussian noise). 
<!-- For Gaussian flow matching, $$\alpha_t = t$$ and $$\sigma_t = 1-t$$. -->

For training, a neural network is estimated to predict $$\hat{\boldsymbol \epsilon} = \hat{\boldsymbol \epsilon}({\bf z}_t; t)$$ that effectively estimates $${\mathbb E} [{\boldsymbol \epsilon} \vert {\bf z}_t]$$, the expected noise added to the data given the noisy sample. Other **model outputs** have been proposed in the literature which are linear combinations of $$\hat{\boldsymbol \epsilon}$$ and $${\bf z}_t$$, and $$\hat{\boldsymbol \epsilon}$$ can be derived from the model output given $${\bf z}_t$$. Learning the model is done by minimizing a weighted mean squared error loss <d-cite key="kingma2024understanding,hoogeboom2024simpler"></d-cite>:

$$
\mathcal{L}(\mathbf{x}) = \mathbb{E}_{t \sim \mathcal{U}(0,1), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \textcolor{green}{w(\lambda_t)} \cdot \frac{d\lambda}{dt} \cdot \|\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\|_2^2 \right],
$$

where $$\textcolor{green}{w(\lambda_t)}$$ is the **weighting function**, balancing the importance of the loss at different noise levels. So far there are three design choices that are seemingly important: noise schedule, model output, and weighting function, that we will elaborate below.




### Noise schedule
The noise schedule of flow matching is in a very simple form: $$\alpha_t = t, \sigma_t = 1 - t$$. Various noise schedules have been proposed in the diffusion literature, such as variance-preserving schedules ($$\alpha_t^2 + \sigma_t^2 = 1$$), variance-exploding schedules ($$\alpha_t=1$$), and other options in between. A few remarks about noise schedule:
1. All different noise schedules can be normalized as a variance-preserving schedule, with a linear scaling of $${\bf z}_t$$ and an unscaling at the model input. The key defining property of a noise schedule is the log signal-to-noise ratio $$\lambda_t$$.
2. The training loss is *invariant* to the training noise schedule, since the loss fuction can be rewritten as $$\mathcal{L}(\mathbf{x}) = \int_{\lambda_{\min}}^{\lambda_{\max}} w(\lambda) \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \|\hat{\boldsymbol{\epsilon}}_\theta - \boldsymbol{\epsilon}\|_2^2 \right] \, d\lambda$$, which is irrelevant to $$\lambda_t$$. However, $$\lambda_t$$ might still affect the variance of the Monte Carlo estimator of the training loss. A few heuristics have been proposed in the literature to automatically adjust the noise schedules over the training course. See [Sander's blog post](https://sander.ai/2024/06/14/noise-schedules.html#adaptive) for a nice summary.
3. As we will see in the next section, the testing noise schedule does impact the sample quality. However, one can choose completely different noise schedules for training and sampling, based on distinct heuristics: For training, it is desirable to have a noise schedule that minimizes the variance of the Monte Calor estimator, whereas for sampling the noise schedule is more related to the discretization error of the ODE / SDE sampling trajectories and the model curvature.



### Model output
Below we summarize several  model outputs proposed in the literature, include a few from diffusion models and the one of flow matching.


| Model Output  | Formulation   | MSE on Model Output  |
| :------------- |:-------------:|-----:|
| $${\boldsymbol \epsilon}$$-prediction      |$$\hat{\boldsymbol \epsilon} $$ | $$\lVert\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\rVert_2^2 $$ |
| $${\bf x}$$-prediction      | $$\hat{\bf x} = \frac{1}{\alpha_t} {\bf z}_t - \frac{\sigma_t}{\alpha_t} \hat{\boldsymbol \epsilon}$$      | $$ \lVert\hat{\bf x} - {\bf x}\rVert_2^2 = e^{- \lambda} \lVert\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\rVert_2^2 $$ |
| $${\bf v}$$-prediction | $$\hat{\bf v} = \alpha_t \hat{\boldsymbol{\epsilon}} - \sigma_t \hat{\bf x} $$      |    $$ \lVert\hat{\bf v} - {\bf v}\rVert_2^2 = \alpha_t^2(e^{-\lambda} + 1)^2 \lVert\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\rVert_2^2 $$ |
| $${\bf u}$$ - flow matching vector field | $$\hat{\bf u} = \hat{\bf x} - \hat{\boldsymbol{\epsilon}} $$      |    $$ \lVert\hat{\bf u} - {\bf u}\rVert_2^2 = (1 + e^{-\lambda / 2})^2 \lVert\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\rVert_2^2 $$ |




### Weighting function
Weighting function is an important design choice for the training loss. 


<div style="background-color: #f9f9f9; border-left: 6px solid #4CAF50; padding: 10px; margin: 10px 0;">

**Note:** This is a remark section. You can customize the colors, text, and padding to suit your webpage's style.

</div>

## Sampling and Straightness Misnomer

<p align="center"><i>"Flow matching paths are straight, whereas diffusion paths are curved."</i></p>

## From Diffusion Models to Flow Matching and back

In this section, we show the equivalence between diffusion models and flow matching approaches from a stochastic process point of view. Note that it is possible to show this equivalence using other apporaches [CITE]

### Flow Matching

In Flow Matching (and stochastic interpolant), we start by defining an interpolation. In order to be consistent with diffusion models notation, we will denote $\mathbf{X}_0 \sim \pi$, where $\pi$ is the data distribution and $\mathbf{X}_1 \sim \mathrm{N}(0, \mathrm{Id})$. 

We start by defining the interpolation

$$
\mathbf{X}_t = \alpha_t \mathbf{X}_0 + \sigma_t \mathbf{X}_1 
$$

We would like to flow from $\mathbf{X}_1$ to $\mathbf{X}_0$.
The associated ODE is given by 

$$
\mathrm{d} \mathbf{X}_t = \{ \dot{\alpha}_t \mathbf{X}_0 + \dot{\sigma}_t \mathbf{X}_1 \} \mathrm{d} t 
$$

Since, we want to flow from $1 \to 0$, we define $\mathbf{Y}_t = \mathbf{X}_{1-t}$ and we get that 

$$
\mathrm{d} \mathbf{Y}_t = - \{ \dot{\alpha}_{1-t} \mathbf{Y}_1 + \dot{\sigma}_{1-t} \mathbf{Y}_0 \} \mathrm{d} t 
$$

Of course, at inference we do not have access to $\mathbf{Y}_1$, i.e., the datapoint and instead replace it by our best guess at time $t$. Hence, we define 

$$
\mathrm{d} \mathbf{Y}_t = - \{ \mathbb{E}[\dot{\alpha}_{1-t} \mathbf{Y}_1 + \dot{\sigma}_{1-t} \mathbf{Y}_0 | \mathbf{Y}_t] \} \mathrm{d} t 
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

### Diffusion models

In diffusion models we usually define 

### How to relate these two models?

Interpolant defines a forward process
Forward process defines an interpolant
By careful identification we can identify the two processes

## Equations

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$



Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) 
that brought a significant improvement to the loading and rendering speed, which is now 
[on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).


## Images and Figures

Its generally a better idea to avoid linking to images hosted elsewhere - links can break and you
might face losing important information in your blog post.
To include images in your submission in this way, you must do something like the following:

```markdown
{% raw %}{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}{% endraw %}
```

which results in the following image:

{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}

To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2025-04-28-[SUBMISSION NAME]` within your submission.

Please avoid using the direct markdown method of embedding images; they may not be properly resized.
Some more complex ways to load images (note the different styles of the shapes/shadows):

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/8.jpg" class="img-fluid z-depth-2" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/10.jpg" class="img-fluid z-depth-2" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/11.jpg" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/12.jpg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/7.jpg" class="img-fluid" %}
    </div>
</div>

### Interactive Figures

Here's how you could embed interactive figures that have been exported as HTML files.
Note that we will be using plotly for this demo, but anything built off of HTML should work
(**no extra javascript is allowed!**).
All that's required is for you to export your figure into HTML format, and make sure that the file
exists in the `assets/html/[SUBMISSION NAME]/` directory in this repository's root directory.
To embed it into any page, simply insert the following code anywhere into your page.

```markdown
{% raw %}{% include [FIGURE_NAME].html %}{% endraw %} 
```

For example, the following code can be used to generate the figure underneath it.

```python
import pandas as pd
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')

fig = px.density_mapbox(
    df, lat='Latitude', lon='Longitude', z='Magnitude', radius=10,
    center=dict(lat=0, lon=180), zoom=0, mapbox_style="stamen-terrain")
fig.show()

fig.write_html('./assets/html/2025-04-28-distill-example/plotly_demo_1.html')
```

And then include it with the following:

```html
{% raw %}<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>{% endraw %}
```

Voila!

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

***

## Code Blocks

This theme implements a built-in Jekyll feature, the use of Rouge, for syntax highlighting.
It supports more than 100 languages.
This example is in C++.
All you have to do is wrap your code in a liquid tag:

{% raw  %}
{% highlight c++ linenos %}  <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers. You can try toggling it on or off yourself below:

{% highlight c++ %}

int main(int argc, char const \*argv[])
{
string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}

{% endhighlight %}

***

## Diagrams

This theme supports generating various diagrams from a text description using [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} plugin.
Below, we generate a few examples of such diagrams using languages such as [mermaid](https://mermaid-js.github.io/mermaid/){:target="\_blank"}, [plantuml](https://plantuml.com/){:target="\_blank"}, [vega-lite](https://vega.github.io/vega-lite/){:target="\_blank"}, etc.

**Note:** different diagram-generation packages require external dependencies to be installed on your machine.
Also, be mindful of that because of diagram generation the first time you build your Jekyll website after adding new diagrams will be SLOW.
For any other details, please refer to [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} README.

**Note:** This is not supported for local rendering! 

The diagram below was generated by the following code:

{% raw %}
```
{% mermaid %}
sequenceDiagram
    participant John
    participant Alice
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
{% endmermaid %}
```
{% endraw %}

{% mermaid %}
sequenceDiagram
participant John
participant Alice
Alice->>John: Hello John, how are you?
John-->>Alice: Great!
{% endmermaid %}

***

## Tweets

An example of displaying a tweet:
{% twitter https://twitter.com/rubygems/status/518821243320287232 %}

An example of pulling from a timeline:
{% twitter https://twitter.com/jekyllrb maxwidth=500 limit=3 %}

For more details on using the plugin visit: [jekyll-twitter-plugin](https://github.com/rob-murray/jekyll-twitter-plugin)

***

## Blockquotes

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>

***


## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body`-sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

***

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behavior, where trailing spaces are not required.)

* Unordered lists can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links. 
http://www.example.com or <http://www.example.com> and sometimes 
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```
 
```python
s = "Python syntax highlighting"
print(s)
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote. 


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.
