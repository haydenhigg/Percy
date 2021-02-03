package percy

func TrainAveragedFromModel(init Model, inputs [][]float64, outputs []float64, iters int, learningRate float64) Model {
	n := float64(len(inputs))

	if n == 0. {
		return init
	}

	weights := makeCopy(init.Weights)
	averages := makeCopy(init.Weights)

	bias := init.Bias
	biasAverage := init.Bias
	
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
			} else {
				for f, w := range weights {
					averages[f] += w
				}
			}

			biasAverage += bias
		}
	}

	for f := range averages {
		averages[f] /= n
	}

	return NewModel(averages, biasAverage / n)
}
