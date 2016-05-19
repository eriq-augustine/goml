package base

import (
   "math"

   "github.com/eriq-augustine/goml/util"
)

// Discretize each feature (column (data point index)) on it's own scale.
// This allows features of diffferent ranges (say a percentage (0 - 1) and
// absolute value (any value (perhaps -1 - 300)) to reside next to each other
// without being put into the same bucket.
func DiscretizeNumericFeatures(data []Tuple, numBuckets int) []Tuple {
   var discreteData []Tuple = make([]Tuple, len(data));
   for i, tuple := range(data) {
      discreteData[i] = tuple.DeepCopy();
   }

   if (len(discreteData) == 0 || numBuckets <= 0) {
      return discreteData;
   }

   for i, _ := range(discreteData[0].Data) {
      DiscretizeNumericFeature(discreteData, numBuckets, i);
   }

   return discreteData;
}

// |data| will get modified.
func DiscretizeNumericFeature(data []Tuple, numBuckets int, featureIndex int) {
   if (len(data) == 0 || len(data[0].Data) < featureIndex || !util.IsNumeric(data[0].Data[featureIndex])) {
      return;
   }

   var min float64 = data[0].NumericValue(featureIndex);
   var max float64 = min;

   for _, tuple := range(data) {
      var val float64 = tuple.NumericValue(featureIndex);
      if (val < min) {
         min = val;
      } else if (val > max) {
         max = val;
      }
   }

   DiscretizeNumericFeatureWithBounds(data, numBuckets, featureIndex, min, max);
}

// |data| will get modified.
func DiscretizeNumericFeatureWithBounds(data []Tuple, numBuckets int, featureIndex int, min float64, max float64) {
   if (numBuckets < 1 || min > max) {
      return;
   }

   var bucketSize float64 = (max - min) / float64(numBuckets);

   for i, _ := range(data) {
      data[i].Data[featureIndex] = Discretize(data[i].NumericValue(featureIndex), min, max, bucketSize, numBuckets);
   }
}

// If (max == min), just put it in the first bucket.
func Discretize(val float64, min float64, max float64, bucketSize float64, numBuckets int) int {
   if (max == min) {
      return 0;
   }

   return int(math.Min(math.Max(val - min, 0) / bucketSize, float64(numBuckets - 1)));
}

// Only works on Numeric values.
// Other values will be left alone.
// Discretized values will start at 0 and go until numBuckets - 1.
func DiscretizeNumericTuple(tuple Tuple, min float64, max float64, numBuckets int) Tuple {
   if (numBuckets < 1 || min > max) {
      return tuple;
   }

   var bucketSize float64 = (max - min) / float64(numBuckets);
   var discreteTuple Tuple = Tuple{make([]interface{}, len(tuple.Data)), tuple.Class};

   for i, val := range(tuple.Data) {
      if (!util.IsNumeric(val)) {
         discreteTuple.Data[i] = tuple.Data[i]
         continue;
      }

      discreteTuple.Data[i] = Discretize(util.NumericValue(val), min, max, bucketSize, numBuckets);
   }

   return discreteTuple;
}
