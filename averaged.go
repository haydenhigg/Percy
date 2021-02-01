package percy

func TrainAveragedFromWeights(initWeights []float64, inps [][]float64, outs []int, iters int, alpha float64) []float64 {
	var weights []float64
	var averages []float64

	if n := len(inps); n == 0 {
		return initWeights
	} else {
		weights = make([]float64, len(initWeights))
		averages = make([]float64, len(initWeights))

		copy(weights, initWeights)
		copy(averages, initWeights)
	}

	inputs := scaleEachToNorm(inps, 1)
	outputs := intsToFloats(outs)

	for iter := 0; iter < iters; iter++ {
		for i, inp := range inputs {
			out := outputs[i]

			if dot(weights, inp) * out <= 0 { // if prediction and output do not match
				for f, w := range weights {
					modW := w + inp[f] * out * alpha
					
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
	
	n := float64(len(inputs))
	
	for f := range averages {
		averages[f] /= n
	}

	return averages
}
