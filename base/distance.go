package base

import (
   "math"
)

type Distancer interface {
   Distance(NumericTuple, NumericTuple) float64
}

type Euclidean struct{}

func (e Euclidean) Distance(x NumericTuple, y NumericTuple) float64 {
   var sum float64 = 0;

   for i := 0; i < x.DataSize(); i++ {
      sum += math.Pow(x.GetNumericData(i) - y.GetNumericData(i), 2);
   }

   return math.Sqrt(sum);
}
