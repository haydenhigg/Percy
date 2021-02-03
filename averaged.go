package percy

func TrainAveragedFromModel(init Model, inputs [][]float64, outputs []float64, iters int, learningRate float64) Model {
	if n := len(inputs); n == 0 {
		return init
	}

	weights := makeCopy(init.Weights)
	averages := makeCopy(init.Weights)

	bias := 0.
	biasAverage := 0.
	
	var out float64
	var gradient float64

	for iter := 0; iter < iters; iter++ {
		for i, inp := range inputs {
			out = outputs[i]

			if (dot(weights, inp) + bias) * out <= 0 { // if prediction and output do not match
				gradient = out * learningRate

				for f := range weights {
					weights[f] += inp[f] * gradient
					averages[f] += weights[f]
				}

				bias += gradient
				biasAverage += bias
			} else {
				for f, w := range weights {
					averages[f] += w
				}

				biasAverage += bias
			}
		}
	}

	return NewModel(averages, biasAverage)
}
