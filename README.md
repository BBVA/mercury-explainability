# mercury-explainability

[![](https://github.com/BBVA/mercury-explainability/actions/workflows/test.yml/badge.svg)](https://github.com/BBVA/mercury-explainability)
![](https://img.shields.io/badge/latest-0.0.3-blue)

***mercury-explainability*** is a library with implementations of different state-of-the-art methods in the field of explainability. They are designed to work efficiently and to be easily integrated with the main Machine Learning frameworks.

## Mercury project at BBVA

Mercury is a collaborative library that was developed by the Advanced Analytics community at BBVA. Originally, it was created as an [InnerSource](https://en.wikipedia.org/wiki/Inner_source) project but after some time, we decided to release certain parts of the project as Open Source.
That's the case with the `mercury-explainability` package. 

The basic block of ***mercury-explainability*** is the `Explainer` class. Each one of the explainers in ***mercury-explainability*** offers a different method for explaining your models and often will return an `Explanation` type object containing the result of that particular explainer.

The usage of most of the explainers you will find in this library follows this schema:

```python
from mercury.explainability import ExplainerExample
explainer = ExplainerExample(function_to_explain)
explanation = explainer.explain(dataset)
```

Basically, you simply need to instantiate your desired `Explainer` (note that the above example `ExplainerExample` does not exist)
providing your custom function you desire to get an explanation for, which usually will be your model’s inference or evaluation function.
These explainers are ready to work efficiently with most of the frameworks you will likely use as a data scientist (yes, included *Spark*).

If you're interested in learning more about the Mercury project, we recommend reading this blog [post](https://www.bbvaaifactory.com/mercury-acelerando-la-reutilizacion-en-ciencia-de-datos-dentro-de-bbva/) from www.bbvaaifactory.com

## User installation

The easiest way to install `mercury-explainability` is using ``pip``:

    pip install -U mercury-explainability

## Help and support 

This library is currently maintained by a dedicated team of data scientists and machine learning engineers from BBVA. 

### Documentation
website: https://bbva.github.io/mercury-explainability/

### Email 
mercury.group@bbva.com
