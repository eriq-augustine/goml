package util

import (
   "math"

   "github.com/eriq-augustine/goml/base"
)

// Only works on Numeric values.
// Other values will be left alone.
// Discretized values will start at 0 and go until numBuckets - 1.
func DiscretizeNumeric(tuple base.Tuple, min float64, max float64, numBuckets int) base.Tuple {
   if (numBuckets < 1 || min > max) {
      return tuple;
   }

   var bucketSize float64 = (max - min) / float64(numBuckets);
   var discreteTuple base.Tuple = base.Tuple{make([]interface{}, len(tuple.Data)), tuple.Class};

   for i, val := range(tuple.Data) {
      if (!IsNumeric(val)) {
         discreteTuple.Data[i] = tuple.Data[i]
         continue;
      }

      discreteTuple.Data[i] = int(math.Min(math.Max(base.NumericValue(val) - min, 0) / bucketSize, float64(numBuckets - 1)));
   }

   return discreteTuple;
}
