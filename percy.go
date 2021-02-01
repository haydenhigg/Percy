package percy

func Train(inps [][]float64, outs []int, iters int, alpha float64) []float64 {
	if n := len(inps); n == 0 {
		return []float64{}
	} else {
		return TrainFromWeights(make([]float64, len(inps[0])), inps, outs, iters, alpha)
	}
}

func TrainAveraged(inps [][]float64, outs []int, iters int, alpha float64) []float64 {
	if n := len(inps); n == 0 {
		return []float64{}
	} else {
		return TrainAveragedFromWeights(make([]float64, len(inps[0])), inps, outs, iters, alpha)
	}
}

func RawPredict(weights, x []float64) float64 {
	return dot(weights, scaleToNorm(x, 1))
}

func Predict(weights, x []float64) int {
	if RawPredict(weights, x) > 0 {
		return 1
	} else {
		return -1
	}
}
