package base

import (
        "reflect"

   "github.com/eriq-augustine/goml/util"
)

type Tuple struct {
	Data  []interface{}
	Class interface{}
}

func (tuple Tuple) Equals(other Tuple) bool {
   return reflect.DeepEqual(tuple, other);
}

// Will panic if bad range.
func (tuple Tuple) NumericValue(featureIndex int) float64 {
   return util.NumericValue(tuple.Data[featureIndex]);
}

func (tuple Tuple) NumericTuple() []float64 {
	var values []float64 = make([]float64, len(tuple.Data))

	for i, value := range tuple.Data {
		values[i] = util.NumericValue(value)
	}

	return values
}

func (tuple Tuple) DeepCopy() Tuple {
   var dataCopy []interface{} = make([]interface{}, len(tuple.Data));
   copy(dataCopy, tuple.Data);
   return Tuple{dataCopy, tuple.Class};
}
