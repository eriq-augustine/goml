package base

import (
	"fmt"
        "reflect"
)

type Tuple struct {
	Data  []interface{}
	Class interface{}
}

func (tuple Tuple) Equals(other Tuple) bool {
   return reflect.DeepEqual(tuple, other);
}

func (tuple Tuple) NumericValue() []float64 {
	var values []float64 = make([]float64, len(tuple.Data))

	for i, value := range tuple.Data {
		values[i] = NumericValue(value)
	}

	return values
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
