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
- `Train(inputs [][]float64, outputs []float64, iters int, alpha float64) []float64`: Trains a Perceptron classifier on the training `inputs` matrix and `outputs` vector for `iters` iterations with a learning rate of `alpha`. Returns the final weights vector.
- `TrainFromWeights(initWeights []float64, inputs [][]float64, outputs []float64, iters int, alpha float64) []float64`: The same as `Train`, but initializes the weights to `initWeights` rather than a zero vector.
- `TrainAveraged(inputs [][]float64, outputs []float64, iters int, alpha float64) []float64`: The same as `Train`, but trains an Averaged Perceptron classifier instead.
- `TrainAveragedFromWeights(initWeights []float64, inputs [][]float64, outputs []float64, iters int, alpha float64) []float64`: The same as `TrainFromWeights`, but trains an Averaged Perceptron classifier instead.
---
- `Predict(weights, x []float64) float64`: Returns the predicted output (which will be {-1, 1}) for the weights vector `weights` and the input vector `x`.
- `RawPredict(weights, x []float64) float64`: Returns the output before being binarized to {-1, 1}.

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
  alpha := 0.01
  
  weights := percy.Train(percy.RegularizeAll(inputs), outputs, iters, alpha)
  
  fmt.Println(percy.Predict(weights, percy.Regularize([]float64{...})))
}
```

## notes

- Assumptions are not checked by this implementation. For example, if each vector of the `inputs` matrix does not have the same length, this will fail; if `inputs` and `outputs` are different lengths, this will fail; if `alpha` is a negative number, the algorithm will not converge; etc.
- Though not necessary, it may be helpful to
  - shuffle the data before training, especially when using the standard Perceptron
  - regularize the norms of all training inputs and of all inputs to be predicted (see `Regularize` and `RegularizeAll`)
  - initialize weights to small random values rather than 0s
