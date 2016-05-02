package base

import (
   "math"
)

type Distancer interface {
   Distance(Tuple, Tuple) float64
}

type Euclidean struct{}

func (e Euclidean) Distance(x Tuple, y Tuple) float64 {
   var sum float64 = 0;

   for i, _ := range(x.Data) {
      sum += math.Pow(float64(NumericValue(x.Data[i]) - NumericValue(y.Data[i])), 2);
   }

   return math.Sqrt(sum);
}
