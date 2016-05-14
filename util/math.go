package util

import (
	"math"
)

const EPSILON = 0.00000001

func FloatEquals(a float64, b float64) bool {
	return math.Abs(a-b) < EPSILON
}

func IsNumeric(obj interface{}) bool {
   if (obj == nil) {
      return false;
   }

   switch obj.(type) {
   case int, int32, int64, uint, uint32, uint64, float32, float64:
      return true;
   }

   return false
}
