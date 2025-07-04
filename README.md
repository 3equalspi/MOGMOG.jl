# MOGMOG

[![Build Status](https://github.com/3equalspi/MOGMOG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/3equalspi/MOGMOG.jl/actions/workflows/CI.yml?query=branch%3Amain)

Autoregressive 3d molecular generator

This project implements a transformer-based autoregressive model for generating small molecules in 3D, inspired by the [Quetzal architecture](https://arxiv.org/abs/2305.15416). It combines*discrete atom-type prediction with continuous position modeling via a Mixture of Gaussians (MoG) in 3D space.

What it does

- Predicts atom types from a vocabulary (e.g. C, N, O, H, F, ..., STOP)
- Models atomic positions using a probabilistic MoG-based diffusion-inspired head
- Builds molecules atom-by-atom, learning chemical structure implicitly

 Architecture

- Embedding layer (e.g. Random Fourier Features)
- Transformer encoder with autoregressive masking
- MoG heads for x,y,z coordinates (position prediction)
- Softmax head for atom type classification

Dataset

- Uses the QM9 dataset, as well as the DLProteinFormats provided by the MurrelGroupRegistry
- Preprocessed molecules are stored in `processed_molecules.jld2` as `Molecule` structs:
  ---julia---
  struct Molecule
      atoms::Vector{String}
      positions::Matrix{Float64}  # (3, L)
  end
