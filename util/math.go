package util

import (
   "fmt"
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
