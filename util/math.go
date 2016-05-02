package util

import (
   "math"
)

const EPSILON = 0.00000001

func FloatEquals(a float64, b float64) bool {
   return math.Abs(a - b) < EPSILON;
}
