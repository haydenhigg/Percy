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
				// writing this loop as well as the averages[f] += weights[f] part in the
				// loop above seems redundant (i.e. it is possible to pull them both out of
				// the if-else and have a separate loop that does it) but it saves one un-
				// necessary loop when prediction and output do not match. miniscule per-
				// formance advantage possibly is not worth the hit to code-readability,
				// but I'm generally prioritising performance here
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
