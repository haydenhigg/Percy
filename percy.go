package percy

func Regularize(arr []float64) []float64 {
	return scaleToNorm(arr, 1)
}

func RegularizeAll(mat [][]float64) [][]float64 {
	return scaleEachToNorm(mat, 1)
}

func Train(inputs [][]float64, outputs []float64, iters int, alpha float64) []float64 {
	if n := len(inputs); n == 0 {
		return []float64{}
	} else {
		return TrainFromWeights(make([]float64, len(inputs[0])), inputs, outputs, iters, alpha)
	}
}

func TrainAveraged(inputs [][]float64, outputs []float64, iters int, alpha float64) []float64 {
	if n := len(inputs); n == 0 {
		return []float64{}
	} else {
		return TrainAveragedFromWeights(make([]float64, len(inputs[0])), inputs, outputs, iters, alpha)
	}
}

func RawPredict(weights, x []float64) float64 {
	return dot(weights, x)
}

func Predict(weights, x []float64) float64 {
	if RawPredict(weights, x) > 0 {
		return 1
	} else {
		return -1
	}
}
