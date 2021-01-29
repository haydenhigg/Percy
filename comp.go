package percy

import "math"

func scaleToNorm(arr []float64, target float64) []float64 {
	s := 0.
	for _, i := range arr { s += i * i }

	fac := target / math.Sqrt(s)

	ret := make([]float64, len(arr))
	for i, x := range arr { ret[i] = x * fac }

	return ret
}

func dot(a, b []float64) float64 {
	s := 0.
	for i, x := range a { s += x * b[i] }

	return s
}
