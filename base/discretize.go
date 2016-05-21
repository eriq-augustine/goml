package base

import (
   "math"

   "github.com/eriq-augustine/goml/util"
)

// Discretize each feature (column (data point index)) on it's own scale.
// This allows features of diffferent ranges (say a percentage (0 - 1) and
// absolute value (any value (perhaps -1 - 300)) to reside next to each other
// without being put into the same bucket.
func DiscretizeNumericFeatures(data []Tuple, numBuckets int) []IntTuple {
   var discreteData []IntTuple = make([]IntTuple, len(data));
   for i, tuple := range(data) {
      // Make a zero'd int tuple.
      discreteData[i] = NewIntTuple(util.InterfaceSlice(make([]int, tuple.DataSize())), tuple.GetClass());
   }

   if (len(data) == 0 || numBuckets <= 0) {
      return discreteData;
   }

   for i := 0; i < data[0].DataSize(); i++ {
      DiscretizeNumericFeature(data, discreteData, numBuckets, i);
   }

   return discreteData;
}

// |data| will get modified.
func DiscretizeNumericFeature(data []Tuple, discreteData []IntTuple, numBuckets int, featureIndex int) {
   if (len(data) == 0 || data[0].DataSize() < featureIndex || !data[0].GetData(featureIndex).IsNumeric()) {
      return;
   }

   var min float64 = (data[0].(NumericTuple)).GetNumericData(featureIndex);
   var max float64 = min;

   for _, tuple := range(data) {
      var val float64 = (tuple.(NumericTuple)).GetNumericData(featureIndex);
      if (val < min) {
         min = val;
      } else if (val > max) {
         max = val;
      }
   }

   DiscretizeNumericFeatureWithBounds(data, discreteData, numBuckets, featureIndex, min, max);
}

// |discreteData| will get modified.
func DiscretizeNumericFeatureWithBounds(data []Tuple, discreteData []IntTuple, numBuckets int, featureIndex int, min float64, max float64) {
   if (numBuckets < 1 || min > max) {
      return;
   }

   var bucketSize float64 = (max - min) / float64(numBuckets);

   for i, _ := range(data) {
      discreteData[i].SetData(featureIndex, Discretize((data[i].(NumericTuple)).GetNumericData(featureIndex), min, max, bucketSize, numBuckets));
   }
}

// If (max == min), just put it in the first bucket.
func Discretize(val float64, min float64, max float64, bucketSize float64, numBuckets int) int {
   if (max == min) {
      return 0;
   }

   return int(math.Min(math.Max(val - min, 0) / bucketSize, float64(numBuckets - 1)));
}
