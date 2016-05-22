package features

import (
   "fmt"

   "github.com/eriq-augustine/goml/base"
)

const (
   NUM_FEATURES = 100
   // TEST
   // NUM_BUCKETS = 100
   NUM_BUCKETS = 3
)

// http://dl.acm.org/citation.cfm?id=1070809
type MRMRReducer struct{}

// TODO(eriq): Don't do anything if too few features.
// It is required that all class labels be present in |data|.
func (reducer MRMRReducer) Init(data []base.Tuple) {
   if (len(data) == 0) {
      panic("Empty training set");
   }

   // TEST
   fmt.Println("--- raw ---");
   fmt.Println(data);

   // Discretize
   var discreteData []base.IntTuple = base.DiscretizeNumericFeatures(data, NUM_BUCKETS);

   // TEST
   fmt.Println("--- discrete ---");
   fmt.Println(discreteData);

   // Marginal Probability (each feature)
   var marginalProbabilities [][]float64 = calcAllMarginalProbabilities(discreteData);

   // TEST
   fmt.Println(marginalProbabilities);

   // Joint Probability (each feature vs (each feature + class)
   var jointProbabilities [][][][]float64 = calcAllJointProbabilities(discreteData);


   // TEST
   printJointProbabilities(jointProbabilities);

   // TODO(eriq);
   // Calc
}

func (reducer MRMRReducer) Reduce(tuple base.Tuple) base.Tuple {
   // TODO(eriq)
   return tuple;
}

// Returns: [featureIndex][bucket (happens to be an index)]marginalProbibility
func calcAllMarginalProbabilities(discreteData []base.IntTuple) [][]float64 {
   var probabilities [][]float64 = make([][]float64, discreteData[0].DataSize());

   for featureIndex := 0; featureIndex < discreteData[0].DataSize(); featureIndex++ {
      probabilities[featureIndex] = calcFeatureMarginalProbabilities(discreteData, featureIndex);
   }

   return probabilities;
}

func calcFeatureMarginalProbabilities(discreteData []base.IntTuple, featureIndex int) []float64 {
   // Note that the data is already zero'd.
   var probabilities []float64 = make([]float64, NUM_BUCKETS);

   for _, tuple := range(discreteData) {
      probabilities[tuple.GetIntData(featureIndex)]++;
   }

   for i, _ := range(probabilities) {
      probabilities[i] /= float64(len(discreteData));
   }

   return probabilities;
}

// Returns: [featureIndex1][featureIndex2][feature1Bucket][feature2Bucket]
// Where featureIndex1 > featureIndex2 (we only need to calc half the solution space since it is symetric).
// Note that featureIndex1 < featureIndex2 is out of bounds and featureIndex1 == featureIndex2 is invalid.
// (We need the later to keep our indexes aligned.)
func calcAllJointProbabilities(discreteData []base.IntTuple) [][][][]float64 {
   var numFeatures int = discreteData[0].DataSize();

   var probabilities [][][][]float64 = make([][][][]float64, numFeatures);

   for featureIndex1 := 0; featureIndex1 < numFeatures; featureIndex1++ {
      probabilities[featureIndex1] = make([][][]float64, featureIndex1 + 1);

      // Invalid.
      probabilities[featureIndex1][featureIndex1] = nil;

      for featureIndex2 := 0; featureIndex2 < featureIndex1; featureIndex2++ {
         probabilities[featureIndex1][featureIndex2] = calcFeatureJointProbabilities(discreteData, featureIndex1, featureIndex2);
      }
   }

   return probabilities;
}

func calcFeatureJointProbabilities(discreteData []base.IntTuple, featureIndex1 int, featureIndex2 int) [][]float64 {
   var probabilities [][]float64 = make([][]float64, NUM_BUCKETS);
   for i := 0; i < NUM_BUCKETS; i++ {
      probabilities[i] = make([]float64, NUM_BUCKETS);
   }

   for _, tuple := range(discreteData) {
      probabilities[tuple.GetIntData(featureIndex1)][tuple.GetIntData(featureIndex2)]++;
   }

   for i, _ := range(probabilities) {
      for j, _ := range(probabilities[i]) {
         probabilities[i][j] /= float64(len(discreteData));
      }
   }

   return probabilities;
}

func printJointProbabilities(jointProbabilities [][][][]float64) {
   for featureIndex1, _ := range(jointProbabilities) {
      if (featureIndex1 == 0) {
         continue;
      }

      fmt.Println("Feature1: ", featureIndex1);

      for featureIndex2, _ := range(jointProbabilities[featureIndex1]) {
         if (featureIndex1 == featureIndex2) {
            continue;
         }

         fmt.Println("   Feature2: ", featureIndex2);

         // Header
         fmt.Print("      1\\2 |   ");
         for i := 0; i < len(jointProbabilities[featureIndex1][featureIndex2]); i++ {
            fmt.Printf("%03d    ", i);
         }
         fmt.Println();

         for feature1Bucket, _ := range(jointProbabilities[featureIndex1][featureIndex2]) {
            fmt.Printf("      %03d |", feature1Bucket);
            for feature2Bucket, _ := range(jointProbabilities[featureIndex1][featureIndex2][feature1Bucket]) {
               fmt.Printf(" %5.4f", jointProbabilities[featureIndex1][featureIndex2][feature1Bucket][feature2Bucket]);
            }
            fmt.Println();
         }
      }
   }
}
