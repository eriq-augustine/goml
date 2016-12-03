package util

import (
   "fmt"
   "math"
)

const EPSILON = 0.00000001

func Sign(val float64) int {
   if (val < 0) {
      return -1;
   }

   return 1;
}

// Does not handle NaN, Inf, -Inf.
func MaxInt(a int, b int) int {
   if (a > b) {
      return a;
   }

   return b;
}

// Does not handle NaN, Inf, -Inf.
func MinInt(a int, b int) int {
   if (a < b) {
      return a;
   }

   return b;
}

func FloatEquals(a float64, b float64) bool {
   return math.Abs(a - b) < EPSILON
}

func FloatToBool(val float64) bool {
   return !FloatEquals(val, 0);
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

func NumericValue(value interface{}) float64 {
   if value == nil {
      return 0
   }

   // Note that we can't group up the cases (fallthrough semantics)
   // since then value would be an interface{} instead of a hard type.
   // Which would then forace a type assertion before the cast.
   switch value := value.(type) {
   case int:
      return float64(value)
   case int32:
      return float64(value)
   case int64:
      return float64(value)
   case uint:
      return float64(value)
   case uint32:
      return float64(value)
   case uint64:
      return float64(value)
   case bool:
      if bool(value) {
         return 1
      }
      return 0
   case float32:
      return float64(value)
   case float64:
      return float64(value)
   case string:
      // TODO: Horner's?
      panic("TODO: String types not yet implemented.")
   default:
      panic(fmt.Sprintf("Unknown type for numeric conversion: %T", value))
   }
}

// The logSumExp trick to prevent over/under-flow..
// The elements of the array need to be exp'd, summed, and then log of the sum needs to be taken.
func LogSumExp(values []float64) float64 {
   _, maxVal := Max(values);

   var sum float64;
   for _, value := range(values) {
      sum += math.Exp(value - maxVal);
   }

   return maxVal + math.Log(sum);
}

func Max(values []float64) (int, float64) {
   if (len(values) == 0) {
      panic("No values sent to max.");
   }

   var maxIndex int = 0;
   var maxValue float64 = values[0];

   for i, value := range(values) {
      if (value > maxValue) {
         maxValue = value;
         maxIndex = i;
      }
   }

   return maxIndex, maxValue;
}

func Sigmoid(val float64) float64 {
   return 1.0 / (1.0 + math.Exp(-1.0 * val));
}
