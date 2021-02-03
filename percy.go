package percy

func Regularize(arr []float64) []float64 {
	return scaleToNorm(arr, 1)
}

func RegularizeAll(mat [][]float64) [][]float64 {
	return scaleEachToNorm(mat, 1)
}

type model struct{
	Weights []float64
	Bias	float64
}

func Train(inputs [][]float64, outputs []float64, iters int, learningRate float64) model {
	if n := len(inputs); n == 0 {
		return model{}
	} else {
		init := model{
			Weights: make([]float64, len(inputs[0])),
			Bias: 0,
		}

		return TrainFromModel(init, inputs, outputs, iters, learningRate)
	}
}

func TrainAveraged(inputs [][]float64, outputs []float64, iters int, learningRate float64) model {
	if n := len(inputs); n == 0 {
		return model{}
	} else {
		init := model{
			Weights: make([]float64, len(inputs[0])),
			Bias: 0,
		}

		return TrainAveragedFromModel(init, inputs, outputs, iters, learningRate)
	}
}

func (mdl model) RawPredict(x []float64) float64 {
	return dot(mdl.Weights, x) + mdl.Bias
}

func (mdl model) Predict(weights, x []float64) float64 {
	if dot(mdl.Weights, x) > -mdl.Bias {
		return 1
	} else {
		return -1
	}
}
