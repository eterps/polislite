# Polislite

A lightweight Pol.is-like.

## Setup

    pip install scikit-learn

## Usage

    python polislite.py

<details><summary>Output</summary>

    Consensus Statements:
    - Climate change requires immediate action (strong agreement)

    Divisive Statements:
    - Nuclear power is necessary for clean energy
    - Carbon tax should be implemented globally
    - Individual actions matter for sustainability
    - Companies should be held liable for emissions

    Group Positions:

    Group 1 characteristics:
    - strongly agrees with: Climate change requires immediate action
    - strongly agrees with: Nuclear power is necessary for clean energy
    - strongly disagrees with: Carbon tax should be implemented globally
    - strongly disagrees with: Individual actions matter for sustainability
    - strongly disagrees with: Companies should be held liable for emissions

    Group 2 characteristics:
    - strongly agrees with: Climate change requires immediate action
    - strongly agrees with: Nuclear power is necessary for clean energy
    - strongly agrees with: Carbon tax should be implemented globally
    - strongly disagrees with: Individual actions matter for sustainability
    - strongly agrees with: Companies should be held liable for emissions

    Group 3 characteristics:
    - strongly agrees with: Climate change requires immediate action
    - strongly disagrees with: Nuclear power is necessary for clean energy
    - strongly agrees with: Carbon tax should be implemented globally
    - strongly agrees with: Individual actions matter for sustainability
    - strongly agrees with: Companies should be held liable for emissions
</details>

## Status

I focused on small incremental improvements through separation of concerns.
In the branch `extract-data-and-output-rendering-to-files`, I separated the
data handling and output rendering. Subsequently, in the branch `extract-lib`,
I isolated the core algorithm into a dedicated library file.
In the branch `plotting` I added a script that plots the user's opinions on a
2D plane.

The full changes can be reviewed in the associated pull requests.

I prefer small steps and improving abstraction before focusing more on
feature completeness for the algorithm.

## Plot example

![Image of a the user's opinions on a 2D plane](plot_example.png)
