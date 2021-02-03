package percy

func TrainFromModel(init model, inputs [][]float64, outputs []float64, iters int, learningRate float64) model {
	if n := len(inputs); n == 0 {
		return init
	}

	weights := makeCopy(init.Weights)

	bias := 0.

	for iter := 0; iter < iters; iter++ {
		for i, inp := range inputs {
			out := outputs[i]

			if (dot(weights, inp) + bias) * out <= 0 { // if prediction and output do not match
				gradient := out * learningRate

				for f := range weights {
					weights[f] += inp[f] * gradient
				}

				bias += gradient
			}
		}
	}

	return model{
		Weights: weights,
		Bias: bias,
	}
}
