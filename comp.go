package percy

import "math"

func norm(arr []float64) float64 {
	s := 0.
	
	for _, i := range arr {
		s += i * i
	}
	
	return math.Sqrt(s)
}

func scaleToNorm(arr []float64, target float64) []float64 {
	if n := norm(arr); n != 0 {
		fac := target / norm(arr)

		ret := make([]float64, len(arr))

		for i, x := range arr {
			ret[i] = x * fac
		}

		return ret
	} else {
		return make([]float64, len(arr))
	}
}

func scaleEachToNorm(mat [][]float64, target float64) [][]float64 {
	ret := make([][]float64, len(mat))

	for i, x := range mat {
		ret[i] = scaleToNorm(x, target)
	}
	
	return ret
}

func intsToFloats(arr []int) []float64 {
	ret := make([]float64, len(arr))
	
	for i, x := range arr {
		ret[i] = float64(x)
	}
	
	return ret
}

func dot(a, b []float64) float64 {
	s := 0.

	for i, x := range a {
		s += x * b[i]
	}

	return s
}
