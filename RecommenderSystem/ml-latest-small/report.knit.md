---
title: "Summary of Collabrative Filtering Methods"
author: "Liu Chaoran"
date: "28/11/2016"
output: 
  html_document:
    toc: true
    toc_depth: 2
    df_print: kable
---

## Overview
A variety methods of CF will be discussed in this blog.

* Gradient Descent CF (GD)
* Stochastic Gradient Descent (SGD)
* Alternating Least Square (ALS)

## Dateset
This example is demonstrated on MovieLens dataset, which consists of 10M rows of user-movie rating records.

## CF models
### Gradient Descent
$$Y = UM$$
where $\textit{Y}$ is the rating matrix with dimension (u,p)
Loss function:
$$Loss = \frac{1}{2}\sum{[R(Y - UM)]^{2}} + \lambda|U|^{2} + \lambda|M|^{2}$$
Gradient function:
$$\frac{dL}{dU} = (Y - UM)M^{T} + \lambda|U|$$ 
