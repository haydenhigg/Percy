package percy

func TrainWithAlpha(rawInputs [][]float64, intOutputs []int, iters int, alpha float64) []float64 {
	inputs := scaleEachToNorm(rawInputs, 1)

	outputs := make([]float64, len(intOutputs))
	for i, o := range intOutputs { outputs[i] = float64(o) }

	var weights []float64
	var averages []float64

	if n := len(inputs); n == 0 {
		return []float64{}
	} else {
		weights = make([]float64, len(inputs[0]))
		averages = make([]float64, len(inputs[0]))
	}

	for iter := 0; iter < iters; iter++ {
		for i, inp := range inputs {
			out := outputs[i]

			if dot(weights, inp) * out <= 0 { // if prediction and output do not match
				for f, w := range weights {
					modW := w + inp[f] * out * alpha // w - Δ
					
					weights[f] = modW
					averages[f] += modW
				}
			} else {
				for f, w := range weights { averages[f] += w }
			}
		}
	}

	return averages
}

func Train(inputs [][]float64, outputs []int, iters int) []float64 {
	return TrainWithAlpha(inputs, outputs, iters, 0.01)
}

func Predict(weights, x []float64) int {
	if dot(weights, scaleToNorm(x, 1)) > 0 { return 1 } else { return -1 }
}