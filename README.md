# percy

Quick, straightforward, and easy-to-use Go implementations of the Perceptron and Averaged Perceptron binary classifiers.

## install

While in your project directory:
`$ git clone https://github.com/haydenhigg/percy`

Then, import it as:
```go
import "./percy"
```

## use

- `Regularize(arr []float64) []float64`: Regularizes the L2 norm of `arr` to 1.
- `RegularizeAll(mat [][]float64) [][]float64`: Regularizes the L2 norm of every row of `mat` to 1.
---
- `NewModel(weights []float64, bias float64) Model`: Creates a model with input `weights` and an input `bias`.
---
- `Train(inputs [][]float64, outputs []float64, iters int, alpha float64) Model`: Trains a Perceptron classifier on the training `inputs` matrix and `outputs` vector for `iters` iterations with a learning rate of `alpha`. Returns the final model (see below).
- `TrainFromModel(init Model, inputs [][]float64, outputs []float64, iters int, alpha float64) Model`: The same as `Train`, but initializes the model to `init` rather than a model with weights as a zero-vector and a bias of zero.
- `TrainAveraged(inputs [][]float64, outputs []float64, iters int, alpha float64) Model`: The same as `Train`, but trains an Averaged Perceptron classifier instead.
- `TrainAveragedFromModel(init Model, inputs [][]float64, outputs []float64, iters int, alpha float64) Model`: The same as `TrainFromWeights`, but trains an Averaged Perceptron classifier instead.
---
- `(mdl Model) Predict(x []float64) float64`: Returns the predicted output (which will be {-1, 1}) for the model `mdl` and the input vector `x`.
- `(mdl Model) RawPredict(x []float64) float64`: Returns the output before being binarized to {-1, 1}.

### Model

The `Model` is just a struct containing the fields `Weights` ([]float64) and `Bias` (float64).

### example

```go
package main

import (
  "fmt"
  "./percy"
)

func main() {
  inputs := [][]float64{[]float64{...}, []float64{...}, ...}
  outputs := []float64{1, -1, ...}
  
  iters := 200
  learningRate := 0.01
  
  trainedModel := percy.Train(percy.RegularizeAll(inputs), outputs, iters, learningRate)
  
  fmt.Println(trainedModel.Predict(percy.Regularize([]float64{...})))
}
```

## notes

- Assumptions are not checked by this implementation. For example, if each vector of the `inputs` matrix does not have the same length, this will fail; if `inputs` and `outputs` are different lengths, this will fail; if `learningRate` is a negative number, the algorithm will not converge; etc.
- Though not necessary, it may be helpful to
  - shuffle the data before training, especially when using the standard Perceptron
  - regularize the norms of all training inputs and of all inputs to be predicted (see `Regularize` and `RegularizeAll`)
  - initialize weights to small random values rather than 0s
