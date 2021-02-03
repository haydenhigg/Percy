package percy

// regularization

func Regularize(arr []float64) []float64 {
	return scaleToNorm(arr, 1)
}

func RegularizeAll(mat [][]float64) [][]float64 {
	return scaleEachToNorm(mat, 1)
}

// models

type model struct{
	Weights []float64
	Bias	float64
}

func NewModel(weights []float64, bias float64) model {
	return model{
		Weights: weights,
		Bias: bias,
	}
}

// training

func Train(inputs [][]float64, outputs []float64, iters int, learningRate float64) model {
	if n := len(inputs); n == 0 {
		return model{}
	} else {
		init := NewModel(make([]float64, len(inputs[0])), 0)

		return TrainFromModel(init, inputs, outputs, iters, learningRate)
	}
}

func TrainAveraged(inputs [][]float64, outputs []float64, iters int, learningRate float64) model {
	if n := len(inputs); n == 0 {
		return model{}
	} else {
		init := NewModel(make([]float64, len(inputs[0])), 0)

		return TrainAveragedFromModel(init, inputs, outputs, iters, learningRate)
	}
}

// prediction

func (mdl model) RawPredict(x []float64) float64 {
	return dot(mdl.Weights, x) + mdl.Bias
}

func (mdl model) Predict(x []float64) float64 {
	if dot(mdl.Weights, x) > -mdl.Bias {
		return 1
	} else {
		return -1
	}
}
