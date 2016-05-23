package features;

import (
   "github.com/eriq-augustine/goml/base"
)

// Trim down the tuples in |tuples| so that they only contain the specified features.
// |features| is a slice of feature indices.
func SelectFeatures(tuples []base.Tuple, features []int) []base.Tuple {
   if (len(features) <= 0) {
      return tuples;
   }

   var rtn []base.Tuple = make([]base.Tuple, len(tuples));
   for tupleIndex, tuple := range(tuples) {
      var data []interface{} = make([]interface{}, len(features));
      for featurePosition, featureIndex := range(features) {
         data[featurePosition] = tuple.GetData(featureIndex);
      }
      rtn[tupleIndex] = base.NewTuple(data, tuple.GetClass());
   }

   return rtn;
}
