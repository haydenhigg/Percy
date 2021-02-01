package percy

func TrainFromWeights(initWeights []float64, inps [][]float64, outs []int, iters int, alpha float64) []float64 {
	var weights []float64

	if n := len(inps); n == 0 {
		return initWeights
	} else {
		weights = make([]float64, len(initWeights))

		copy(weights, initWeights)
	}

	inputs := scaleEachToNorm(inps, 1)
	outputs := intsToFloats(outs)

	for iter := 0; iter < iters; iter++ {
		for i, inp := range inputs {
			out := outputs[i]

			if dot(weights, inp) * out <= 0 { // if prediction and output do not match
				for f := range weights {
					weights[f] += inp[f] * out * alpha
				}
			}
		}
	}

	return weights
}
