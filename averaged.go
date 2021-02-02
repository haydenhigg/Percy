package percy

func TrainAveragedFromWeights(initWeights []float64, inputs [][]float64, outputs []float64, iters int, learningRate float64) []float64 {
	var weights []float64
	var averages []float64

	if n := len(inputs); n == 0 {
		return initWeights
	} else {
		weights = make([]float64, len(initWeights))
		averages = make([]float64, len(initWeights))

		copy(weights, initWeights)
		copy(averages, initWeights)
	}

	for iter := 0; iter < iters; iter++ {
		for i, inp := range inputs {
			out := outputs[i]

			if dot(weights, inp) * out <= 0 { // if prediction and output do not match
				for f, w := range weights {
					modW := w + inp[f] * out * learningRate
					
					weights[f] = modW
					averages[f] += modW
				}
			} else {
				for f, w := range weights {
					averages[f] += w
				}
			}
		}
	}

	return averages
}
