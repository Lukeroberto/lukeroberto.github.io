---
title: Peak Finding with Wavelets
date: 2025-07-06
is_post: true
---

I have recently been exploring the use of wavelets for use at work and I figured I would write a bit of a dive into what
they are and how they can be used. Be warned that this post will contain some math and code that will be used to convey
the ideas.


## What is a wavelet?

A wavelet is a signal similar to that of a sine or a cosine.  They are signals that are a short, oscillating burst that
end at zero on both ends. The more formal requirements to be considered a wavelet is that they need to integrate to
zero, and should be localized in time (zero outside a certain domain). 

The main difference is that sines and cosines are infinitely repeating signals that violate that second requirement.
This also ends up being one of the major limitations of the fourier transform. It is unable to extract frequencies at
particular instances in a signal. 

There are a variety of wavelet types, engineered for a myriad of different applications. Some can be used for denoising
and peak finding, and 


PHOTOS OF WAVELETS

## What is wavelet analysis?

Wavelet analysis is similar to fourier analysis. Fourier analysis seeks to represent a signal in terms of its
constituent sines and cosines, whereas the wavelet analysis will decompose a signal into its constituent wavelets scaled
and shifted. So if there is a short-lived frequency component in the signal, the fourier description of the signal would
only be able to tell you that the frequency was present, but not where/when in the signal. Due to the short-time nature
of a wavelet, this gives us the ability to discern this. It will return a decomposition of the signal that includes the
frequency (or scale as it’s more commonly know) content at each sample along the signal.

Similar to the existence of the fourier transform, there is an equivalent wavelet transform that can be used to convert
a signal into its wavelet-decomposed form. This can be thought of as a dot-product of the signal with each wavelet at
each time/scale bin.


## Applications: Denoising and Peak finding

The main applications that I have experience with wavelet analysis is that of denoising and peak finding. 

Denoising is the act of taking signal and removing noise that has corrupted it in any way. This can happen all along the
way of data collection. The goal is to extract the features of the signal that are most meaningful and carry the
information that you seek to extract. Wavelets are a prime usecase for this application because one can use the
wavelet-reconstruction of the signal to observe the features at different length scales at different points in the
signal.

Peak finding is the act of finding local maxima within a one dimensional signal (potentially higher dimensional).  This
is different than taking the argmax of a signal, for 2 reasons. The first one being that there might be multiple peaks
we are looking for, so the argmax approach would fail as the “next” largest value with most likely be directly adjacent
to the current peak. The other issue is that natural signals tend to be corrupted w/ noise and so any peak finding
approach either needs to denoise or use features for peak selection that are robust to short scale fluctuations in a
signal. 

A more general approach to peak finding is to use



##  References:

O’haver, T. (2025). Intro to Signal Processsing. https://terpconnect.umd.edu/~toh/spectrum/index.html

