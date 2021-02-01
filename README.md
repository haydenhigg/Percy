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

- `Train(inputs [][]float64, outputs []int, iters int, alpha float64) []float64`: Trains a Perceptron classifier on the training `inputs` matrix and `outputs` vector (where each value of `outputs` is {-1, 1}) for `iters` iterations with a learning rate of `alpha`. Returns the final weights vector.
- `TrainFromWeights(initWeights []float64, inputs [][]float64, outputs []int, iters int, alpha float64) []float64`: The same as `Train`, but initializes the weights to `initWeights` rather than a zero vector.
- `TrainAveraged(inputs [][]float64, outputs []int, iters int, alpha float64) []float64`: The same as `Train`, but trains an Averaged Perceptron classifier instead.
- `TrainAveragedFromWeights(initWeights []float64, inputs [][]float64, outputs []int, iters int, alpha float64) []float64`: The same as `TrainFromWeights`, but trains an Averaged Perceptron classifier instead.
- `Predict(weights, x []float64) int`: Returns the predicted output (which will be {-1, 1}) for the weights vector `weights` and the input vector `x`.
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
  outputs := []int{1, -1, ...}
  
  iters := 200
  alpha := 0.01
  
  weights := percy.Train(inputs, outputs, iters, alpha)
  
  fmt.Println(percy.Predict(weights, []float64{...}))
}
```

## notes

- Assumptions are not checked by this implementation. For example, if each vector of the `inputs` matrix does not have the same length, this will fail; if `inputs` and `outputs` are different lengths, this will fail; if `alpha` is a negative number, the algorithm will not converge; etc.
- Internally, all inputs passed to `Train`/`TrainFromWeights`/`TrainAveraged`/`TrainAveragedFromWeights` (as `inputs`) and `Predict`/`RawPredict` (as `x`) are scaled to have L2 norm = 1.
- It may be helpful, in many cases, to shuffle the data before training, especially when using the standard Perceptron.
