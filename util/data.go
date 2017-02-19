package util

import (
   "math/rand"
)

const (
   DEFAULT_FAKE_DATA_NUM_POINTS = 200
   DEFAULT_FAKE_DATA_NUM_CLASSES = 2
   DEFAULT_FAKE_DATA_NUM_FEATURES = 2
   DEFAULT_FAKE_DATA_MAX_DISTANCE_TO_CENTER = 30.0
   DEFAULT_FAKE_DATA_RANGE_MIN = -100.0
   DEFAULT_FAKE_DATA_RANGE_MAX = 100.0
)

// TODO(eriq): Dest with float64 inseatd of float64.

func FakeDataDefault() ([][]float64, []int) {
   return FakeData(0, 0, 0, 0, nil, nil, rand.Int63());
}

// Pass zero/negative or nil to any param to default (except seed).
func FakeData(numPoints int, numClasses int, numFeatures int, maxDistanceToCenter float64,
              rangeMin []float64, rangeMax []float64, seed int64) ([][]float64, []int) {
   var random *rand.Rand = rand.New(rand.NewSource(seed));

   if (numPoints <= 0) {
      numPoints = DEFAULT_FAKE_DATA_NUM_POINTS;
   }

   if (numClasses <= 0) {
      numClasses = DEFAULT_FAKE_DATA_NUM_CLASSES;
   }

   if (numFeatures <= 0) {
      numFeatures = DEFAULT_FAKE_DATA_NUM_FEATURES;
   }

   if (maxDistanceToCenter <= 0) {
      maxDistanceToCenter = DEFAULT_FAKE_DATA_MAX_DISTANCE_TO_CENTER;
   }

   if (rangeMin == nil || len(rangeMin) == 0) {
      rangeMin = make([]float64, numFeatures);
      for i := 0; i < numFeatures; i++ {
         rangeMin[i] = DEFAULT_FAKE_DATA_RANGE_MIN;
      }
   }

   if (len(rangeMin) != numFeatures) {
      panic("Size of rangeMin must match the number of features.");
   }

   if (rangeMax == nil || len(rangeMax) == 0) {
      rangeMax = make([]float64, numFeatures);
      for i := 0; i < numFeatures; i++ {
         rangeMax[i] = DEFAULT_FAKE_DATA_RANGE_MAX;
      }
   }

   if (len(rangeMax) != numFeatures) {
      panic("Size of rangeMax must match the number of features.");
   }

   return fakeData(numPoints, numClasses, numFeatures, maxDistanceToCenter, rangeMin, rangeMax, random);
}

func fakeData(numPoints int, numClasses int, numFeatures int, maxDistanceToCenter float64,
              rangeMin []float64, rangeMax []float64, random *rand.Rand) ([][]float64, []int) {
   var data [][]float64 = make([][]float64, numPoints);
   var classes []int = make([]int, numPoints);

   // Calculate the centers for each class.
   // Just evenly space them in the numeric space for each feature.
   var centers [][]float64 = make([][]float64, numClasses);
   for i := 0; i < numClasses; i++ {
      var center []float64 = make([]float64, numFeatures);

      for j := 0; j < numFeatures; j++ {
         var step float64 = (rangeMax[j] - rangeMin[j]) / (float64(numClasses) + 1.0);

         center[j] = rangeMin[j] + float64(i + 1) * step;
      }

      centers[i] = center;
   }

   for i := 0; i < numPoints; i++ {
      var features []float64 = make([]float64, numFeatures);
      var class int = rand.Intn(numClasses);

      for j := 0; j < numFeatures; j++ {
         features[j] = centers[class][j] + (random.Float64() * maxDistanceToCenter * 2.0) - maxDistanceToCenter;
      }

      data[i] = features;
      classes[i] = class;
   }

   return data, classes;
}
