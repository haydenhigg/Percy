package perceptron

func TrainWithAlpha(inputs [][]float64, outputs []int, iters int, alpha float64) []float64 {
	var weights []float64
	var averages []float64

	if n := len(inputs); n == 0 {
		return []float64{}
	} else {
		weights = make([]float64, len(inputs[0]))
		averages = make([]float64, len(inputs[0]))
	}

	for iter := 0; iter < iters; iter++ {
		for i, x := range inputs {
			inp := scaleToNorm(x, 1)
			out := float64(outputs[i])

			if dot(weights, inp) * out <= 0 { // if prediction and output do not match
				for i, w := range weights {
					modW := w + inp[i] * out * alpha // w - Δ
					
					weights[i] = modW
					averages[i] += modW
				}
			} else {
				for i, w := range weights { averages[i] += w }
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